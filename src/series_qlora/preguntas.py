"""Preguntas en lenguaje natural a partir de las especificaciones: prompt, validación por código y marcas de calidad.

Un modelo (Claude Haiku) redacta dos versiones de cada pregunta (coloquial y técnica); todo lo demás es código:
el período que debe aparecer, las reglas que debe cumplir el texto y las marcas con las que se arma el conjunto final.
`llamar` es inyectable (callable(prompt, max_tokens) -> (texto, truncado, tokens_entrada, tokens_salida)).
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable

import pandas as pd

from .texto import norm

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
TRIM = ["primer", "segundo", "tercer", "cuarto"]
PAT_INTERANUAL = (r"interanual|\bano (anterior|pasado|previo)|mismo (mes|trimestre|periodo)|"
                  r"un ano (antes|atras)|hace un ano")
RE_MES = re.compile(r"\b(" + "|".join(MESES + ["setiembre"]) + r")\b")
RE_TRIM = re.compile(r"\btrimestre\b")
RE_TASA_UNIDAD = re.compile(r"%|porcent|variaci")
RE_VAR_TITULO = re.compile(r"(^|_)(var|variacion|ia)(_|$)")

# Precios de Claude Haiku (USD por millón de tokens) con los que se declaró el costo
PRECIO_ENTRADA, PRECIO_SALIDA = 0.10, 0.50


def costo_usd(entrada: int, salida: int) -> float:
    return entrada * PRECIO_ENTRADA / 1e6 + salida * PRECIO_SALIDA / 1e6


def _texto(v) -> str:
    return "" if pd.isna(v) else str(v)


# ---------- período ----------
def periodo_de_spec(r) -> str | None:
    """Período que debe nombrar la pregunta ('2021-03' o '2021'), o None si no lleva fecha (ultimo_dato)."""
    if r["tipo"] == "ultimo_dato":
        return None
    p = json.loads(r["params"])
    if r["tipo"] == "variacion_interanual":
        e = p["end_date"]
        return e[:4] if r["frecuencia"] == "anual" else e[:7]
    return p["start_date"]


def periodo_humano(frecuencia: str, p: str) -> str:
    """'2021-03' -> 'marzo de 2021' (mensual) o 'primer trimestre de 2021' (trimestral); anual: '2021'."""
    if frecuencia == "anual":
        return p[:4]
    y, m = p[:4], int(p[5:7])
    if frecuencia == "mensual":
        return f"{MESES[m - 1]} de {y}"
    return f"{TRIM[(m - 1) // 3]} trimestre de {y}"


# ---------- validación por código ----------
def validar(r, q, estilo: str, p: str | None) -> str:
    """Motivo por el que la pregunta no sirve, o '' si pasa. Se corre antes de aceptar lo que escribe el modelo."""
    if not isinstance(q, str) or not q.strip():
        return "vacía"
    n = len(q.split())
    lo, hi = (3, 14) if estilo == "coloquial" else (8, 45)
    if not lo <= n <= hi:
        return f"largo ({n} palabras)"
    t = norm(q)
    anios = {int(a) for a in re.findall(r"\b(?:19|20)\d{2}\b", t)}
    if p is None:
        if re.search(r"\d", t):
            return "tiene números y no debería"
    else:
        y = int(p[:4])
        permitidos = {y, y - 1} if r["tipo"] == "variacion_interanual" else {y}
        if y not in anios:
            return "falta el año"
        if not anios <= permitidos:
            return "año de más"
        if re.search(r"\d", re.sub(r"\b(?:19|20)\d{2}\b", "", t)):
            return "tiene otros números"
        if r["frecuencia"] == "mensual":
            m = int(p[5:7])
            nombres = [MESES[m - 1]] + (["setiembre"] if m == 9 else [])
            if not any(x in t for x in nombres):
                return "falta el mes"
        if r["frecuencia"] == "trimestral" and "trimestre" not in t:
            return "falta 'trimestre'"
    titulo = norm(r["serie_titulo"])
    umbral = 3 if estilo == "coloquial" else 5
    if len(titulo.split()) >= umbral and titulo in t:
        return "copia el título"
    if r["tipo"] == "variacion_interanual" and not re.search(PAT_INTERANUAL, t):
        return "no aclara que es interanual"
    return ""


def defecto_periodo(frecuencia: str, tipo: str, texto) -> str:
    """Período que no corresponde a la serie (p. ej. un mes en una serie anual); lo ve el verificador de fidelidad
    solo por sentido, así que esta regla lo atrapa por código."""
    t = norm(texto)
    if tipo == "ultimo_dato" and RE_MES.search(t):
        return "fecha en pregunta de último dato"
    if frecuencia == "anual" and (RE_MES.search(t) or RE_TRIM.search(t)):
        return "mes o trimestre en serie anual"
    if frecuencia == "trimestral" and RE_MES.search(t):
        return "mes en serie trimestral"
    if frecuencia == "mensual" and RE_TRIM.search(t):
        return "trimestre en serie mensual"
    return ""


def marca_var(tipo: str, titulo, unidades) -> str:
    """Una variación interanual sobre una serie que ya es una tasa o un porcentaje se marca, no se descarta:
    'var_sobre_tasa' (la serie ya es una variación) o 'var_sobre_porcentaje' (la serie está en %)."""
    if tipo != "variacion_interanual":
        return ""
    tit, uni = str(titulo).lower(), str(unidades).lower()
    if RE_VAR_TITULO.search(tit) or "variaci" in uni:
        return "var_sobre_tasa"
    if RE_TASA_UNIDAD.search(uni) or "tasa" in tit:
        return "var_sobre_porcentaje"
    return ""


# ---------- prompt y generación ----------
def armar_prompt(r, p: str | None, serie) -> str:
    """Prompt que pide las dos versiones de la pregunta. `serie` es la fila del catálogo de la serie."""
    if r["tipo"] == "ultimo_dato":
        pedido = "el dato más reciente disponible."
        regla = "No menciones ninguna fecha, año ni período."
    else:
        hp = periodo_humano(r["frecuencia"], p)
        if r["tipo"] == "variacion_interanual":
            pedido = f"cuánto varió respecto del mismo período del año anterior (variación interanual) en {hp}."
            regla = (f"Mencioná este período: {hp}. Escribí el año en cifras y el mes o el trimestre en palabras. "
                     "Podés mencionar también el año anterior. En ambas versiones dejá explícito que es una "
                     "variación interanual o respecto del año anterior.")
        else:
            pedido = f"el valor del dato en {hp}."
            regla = (f"Mencioná este período: {hp}. Escribí el año en cifras y el mes o el trimestre en palabras. "
                     "No menciones ningún otro año.")
    return f"""Sos un usuario argentino que quiere consultar un dato estadístico y no conoce los nombres técnicos de la base de datos.
Escribí DOS versiones de la misma pregunta:
- "coloquial": entre 4 y 12 palabras, como escribiría alguien rápido en un chat. Solo el tema principal, sin detalles técnicos.
- "tecnica": entre 10 y 35 palabras, más precisa, con el detalle necesario para distinguir la serie de otras parecidas (frecuencia, unidad, base u otra característica).

La pregunta pide: {pedido}

Reglas:
- {regla}
- No agregues otros números.
- No copies el título técnico ni uses guiones bajos.

Devolvé solo JSON: {{"coloquial": "...", "tecnica": "..."}}

Datos de la serie:
Título técnico: {_texto(serie['serie_titulo'])}
Descripción: {_texto(serie['serie_descripcion'])}
Unidades: {_texto(serie['serie_unidades'])}
Frecuencia: {_texto(serie['frecuencia'])}
Dataset: {_texto(serie['dataset_titulo'])}
Distribución: {_texto(serie['distribucion_titulo'])}"""


def generar_pregunta(r, serie, llamar: Callable, intentos: int = 3) -> dict:
    """Pide la pregunta y la acepta solo si pasa `validar`; reintenta hasta `intentos` veces.

    El último intento sube el tope de tokens (un corte por max_tokens se cuenta como rechazo). Los errores de la
    llamada (red, clave) devuelven una fila con ok=False y el error; los fatales se tratan en el script.
    """
    p = periodo_de_spec(r)
    prompt = armar_prompt(r, p, serie)
    motivos, ent, sal = [], 0, 0
    for intento in range(intentos):
        mt = 3000 if intento < intentos - 1 else 6000
        try:
            txt, truncado, e, s = llamar(prompt, mt)
        except Exception as ex:
            return {"ok": False, "coloquial": "", "tecnica": "", "motivos": " | ".join(motivos),
                    "error": f"{type(ex).__name__}: {str(ex)[:100]}", "intentos": intento + 1,
                    "ent": ent, "sal": sal}
        ent += e
        sal += s
        if truncado:
            motivos.append("max_tokens")
            continue
        try:
            js = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            col, tec = js["coloquial"], js["tecnica"]
        except Exception:
            motivos.append("JSON inválido")
            continue
        m1, m2 = validar(r, col, "coloquial", p), validar(r, tec, "tecnica", p)
        if not m1 and not m2:
            return {"ok": True, "coloquial": col, "tecnica": tec, "motivos": " | ".join(motivos), "error": "",
                    "intentos": intento + 1, "ent": ent, "sal": sal}
        if m1:
            motivos.append(f"coloquial: {m1}")
        if m2:
            motivos.append(f"tecnica: {m2}")
    return {"ok": False, "coloquial": "", "tecnica": "", "motivos": " | ".join(motivos),
            "error": f"sin respuesta válida tras {intentos} intentos", "intentos": intentos, "ent": ent, "sal": sal}


# ---------- verificación de fidelidad (otra llamada al modelo, con su propio prompt) ----------
def operacion(r) -> str:
    p = periodo_de_spec(r)
    if r["tipo"] == "ultimo_dato":
        return "el dato más reciente disponible"
    hp = periodo_humano(r["frecuencia"], p)
    return f"la variación interanual en {hp}" if r["tipo"] == "variacion_interanual" else f"el valor en {hp}"


def prompt_fidelidad(r, serie) -> str:
    return f"""Verificá si cada pregunta pide el dato descrito abajo, o si cambia su sentido (otra medida, otra entidad, otra operación u otro período).
Una pregunta coloquial puede ser más vaga que el dato, pero tiene que poder referirse razonablemente a ESE dato y no a uno claramente distinto.
Marcá "ok": false solo si hay un cambio de sentido real.

Dato real:
- Título técnico: {_texto(serie['serie_titulo'])}
- Descripción: {_texto(serie['serie_descripcion'])}
- Unidades: {_texto(serie['serie_unidades'])}
- Frecuencia: {_texto(serie['frecuencia'])}
- Dataset: {_texto(serie['dataset_titulo'])}
- Distribución: {_texto(serie['distribucion_titulo'])}
- Lo que se pide: {operacion(r)}

Pregunta coloquial: {r['coloquial']}
Pregunta técnica: {r['tecnica']}

Devolvé solo JSON: {{"coloquial": {{"ok": true, "motivo": ""}}, "tecnica": {{"ok": true, "motivo": ""}}}}"""


def leer_fidelidad(texto: str) -> dict:
    """Extrae las cuatro columnas col_ok, col_motivo, tec_ok, tec_motivo de la respuesta JSON del modelo."""
    js = json.loads(texto[texto.find("{"): texto.rfind("}") + 1])
    return {"col_ok": bool(js["coloquial"]["ok"]), "col_motivo": str(js["coloquial"].get("motivo", "")),
            "tec_ok": bool(js["tecnica"]["ok"]), "tec_motivo": str(js["tecnica"].get("motivo", ""))}


# ---------- armado final (sin costo) ----------
def armar_final(preguntas: pd.DataFrame, fidelidad: pd.DataFrame, catalogo: pd.DataFrame,
                exigir_verificacion: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une preguntas y verificación de fidelidad y aplica las reglas de exclusión; devuelve (finales, excluidas).

    - excluida: la versión técnica cambia el sentido del dato (o no fue verificada), o tiene un defecto de período.
    - coloquial_ambigua: el coloquial cambia el sentido o está repetido en el conjunto (no se usa para entrenar ni
      para medir; la versión técnica de la misma fila sí).
    - marca_var: variación interanual sobre una serie que ya es tasa o porcentaje (ver `marca_var`).
    `catalogo` está indexado por serie_id. La salida conserva las columnas de `preguntas` más las marcas.
    """
    M = preguntas.merge(fidelidad[["spec_id", "col_ok", "col_motivo", "tec_ok", "tec_motivo"]],
                        on="spec_id", how="left")
    M["verificada"] = M["col_ok"].notna() & M["tec_ok"].notna()
    if exigir_verificacion:
        assert M["verificada"].all(), "Faltan verificaciones de fidelidad"
    M["col_ok"] = M["col_ok"].eq(True)
    M["tec_ok"] = M["tec_ok"].eq(True)
    M["dup_coloquial"] = M["coloquial"].str.lower().str.strip().duplicated(keep=False)
    M["coloquial_ambigua"] = ~M["col_ok"] | M["dup_coloquial"]
    ref = catalogo[["serie_titulo", "serie_unidades"]].add_prefix("cat_")
    M = M.merge(ref, left_on="serie_id", right_index=True, how="left")
    M["def_periodo"] = [defecto_periodo(f, t, c) or defecto_periodo(f, t, te)
                        for f, t, c, te in zip(M["frecuencia"], M["tipo"], M["coloquial"], M["tecnica"], strict=True)]
    M["marca_var"] = [marca_var(t, ti, u) for t, ti, u in zip(M["tipo"], M["cat_serie_titulo"],
                                                               M["cat_serie_unidades"], strict=True)]
    M["excluida"] = ~M["verificada"] | ~M["tec_ok"] | (M["def_periodo"] != "")
    sobran = ["cat_serie_titulo", "cat_serie_unidades", "def_periodo", "verificada"]
    return M[~M["excluida"]].drop(columns=sobran).copy(), M[M["excluida"]].drop(columns=sobran).copy()

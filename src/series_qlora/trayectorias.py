"""Trayectorias de entrenamiento: pregunta -> search_series -> get_series -> submit_answer.

Cada trayectoria es una conversación completa con resultados reales de las herramientas (el índice de búsqueda y la
respuesta cruda de la API que se guardó al construir las preguntas). Solo el texto de la consulta de búsqueda lo
escribe un modelo (Claude Haiku); todo lo demás sale de código y de datos.
"""
from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd

from .evaluador import periodo

PROMPT_CONSULTA = """Un usuario le hizo esta pregunta a un asistente que consulta un catálogo de series estadísticas argentinas.
Escribí la consulta de búsqueda (entre 2 y 7 palabras, sin fechas ni números) que conviene teclear en el buscador del catálogo, que busca por palabras en títulos y descripciones.
Usá las palabras clave del tema de la pregunta. Podés agregar un término técnico equivalente si es habitual en estadísticas oficiales, pero no copies el título técnico.

Pregunta: {pregunta}

Dato al que se refiere (solo para orientarte):
Título: {titulo} | Descripción: {descripcion} | Dataset: {dataset}

Devolvé solo JSON: {{"consulta": "..."}}"""


def _texto(v) -> str:
    return "" if pd.isna(v) else str(v)


def preguntas_de_entrenamiento(specs: pd.DataFrame, finales: pd.DataFrame, pct_sin_datos=0.18) -> pd.DataFrame:
    """Preguntas candidatas: una por estilo (técnica y, si no es ambigua, coloquial).

    - Se descartan las variaciones calculadas sobre una tasa (`var_sobre_tasa`).
    - Las preguntas sin datos solo van en estilo técnico y se submuestrean hasta ser ~`pct_sin_datos` del total,
      para que el modelo no aprenda a responder "no hay datos" por defecto.
    """
    E = specs.copy()
    E["spec_id"] = E.index
    F = finales[finales["marca_var"].fillna("") != "var_sobre_tasa"]
    F = F.merge(E[["spec_id", "data", "respuesta_final", "serie_titulo", "serie_unidades"]], on="spec_id", how="left")
    filas = []
    for r in F.to_dict("records"):
        filas.append({**r, "estilo": "tecnica", "pregunta": r["tecnica"]})
        if not r["coloquial_ambigua"]:
            filas.append({**r, "estilo": "coloquial", "pregunta": r["coloquial"]})
    Q = pd.DataFrame(filas)
    con_datos = Q[Q["tipo"] != "sin_datos"]
    sin_datos = Q[(Q["tipo"] == "sin_datos") & (Q["estilo"] == "tecnica")]
    n_sin = int(round(pct_sin_datos / (1 - pct_sin_datos) * len(con_datos)))
    sin_datos = sin_datos.sample(n=min(n_sin, len(sin_datos)), random_state=3)
    Q = pd.concat([con_datos, sin_datos]).sample(frac=1, random_state=5).reset_index(drop=True)
    Q["qid"] = Q["spec_id"].astype(str) + "-" + Q["estilo"]
    return Q


# ---------- consulta de búsqueda (la única parte que escribe un modelo) ----------
def prompt_consulta(pregunta: str, serie: pd.Series) -> str:
    return PROMPT_CONSULTA.format(
        pregunta=pregunta, titulo=_texto(serie["serie_titulo"]), descripcion=_texto(serie["serie_descripcion"]),
        dataset=_texto(serie["dataset_titulo"]))


def extraer_consulta(texto: str) -> str | None:
    """Consulta válida (1 a 8 palabras, sin dígitos) dentro de la respuesta JSON del modelo, o None."""
    try:
        q = json.loads(texto[texto.find("{"): texto.rfind("}") + 1])["consulta"].strip()
    except Exception:
        return None
    return q if 1 <= len(q.split()) <= 8 and not re.search(r"\d", q) else None


def pedir_consulta(cliente, fila: dict, catalogo: pd.DataFrame, modelo="claude-haiku-5-5",
                   intentos=3, errores_fatales: tuple = ()) -> dict:
    """Pide la consulta a Claude (sin razonamiento). `cliente` es un anthropic.Anthropic o un doble de prueba.

    Devuelve {'qid', 'ok', 'error', 'consulta', 'ent', 'sal'}. Los errores de `errores_fatales` (p. ej. clave
    inválida) se relanzan para cortar toda la corrida; cualquier otro error queda registrado en la fila.
    """
    prompt = prompt_consulta(fila["pregunta"], catalogo.loc[fila["serie_id"]])
    try:
        for _ in range(intentos):
            msg = cliente.messages.create(model=modelo, max_tokens=200, messages=[{"role": "user", "content": prompt}],
                                          extra_body={"thinking": {"type": "disabled"}})
            texto = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            q = extraer_consulta(texto)
            if q:
                return {"qid": fila["qid"], "ok": True, "error": "", "consulta": q,
                        "ent": msg.usage.input_tokens, "sal": msg.usage.output_tokens}
        return {"qid": fila["qid"], "ok": False, "error": "sin consulta válida"}
    except errores_fatales:
        raise
    except Exception as e:
        return {"qid": fila["qid"], "ok": False, "error": f"{type(e).__name__}: {str(e)[:100]}"}


def costo_usd(resultados: pd.DataFrame, entrada_por_millon=0.10, salida_por_millon=0.50) -> float:
    return (resultados["ent"].fillna(0).sum() * entrada_por_millon + resultados["sal"].fillna(0).sum() * salida_por_millon) / 1e6


# ---------- armado de la trayectoria ----------
def argumentos_submit(fila: dict, datos: list, params: dict) -> dict:
    """Respuesta final que el modelo debe aprender a entregar, según el tipo de pregunta."""
    tipo = fila["tipo"]
    if tipo == "sin_datos":
        texto = re.sub(r"\s*La serie tiene datos entre.*$", "", fila["respuesta_final"])
        return {"respuesta": texto, "valor": None, "fecha": None}
    if tipo == "ultimo_dato":
        return {"respuesta": fila["respuesta_final"], "valor": datos[0][1], "fecha": datos[0][0]}
    if tipo == "valor_puntual":
        return {"respuesta": fila["respuesta_final"], "valor": datos[0][1], "fecha": params["start_date"]}
    # variación interanual: la API devuelve una fracción y la respuesta va en porcentaje
    return {"respuesta": fila["respuesta_final"], "valor": round(datos[-1][1] * 100, 2),
            "fecha": periodo(datos[-1][0], fila["frecuencia"])}


def _llamada(i: int, nombre: str, argumentos: dict) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": nombre, "arguments": argumentos}}]}


def armar_trayectoria(fila: dict, consulta: str, candidatos_json: str) -> dict | None:
    """Conversación completa, o None si la serie objetivo no está entre los candidatos devueltos por el buscador.

    `candidatos_json` es lo que devolvió `Herramientas.search_series(consulta)`: se usa tal cual como respuesta
    de la herramienta, de modo que el modelo ve en entrenamiento exactamente lo que verá en la evaluación.
    """
    candidatos = json.loads(candidatos_json)
    ids = [c["serie_id"] for c in candidatos]
    if fila["serie_id"] not in ids:
        return None
    datos, params = json.loads(fila["data"]), json.loads(fila["params"])
    respuesta_get = {"status": 200, "serie_id": fila["serie_id"], "titulo": fila["serie_titulo"],
                     "unidades": None if pd.isna(fila["serie_unidades"]) else fila["serie_unidades"],
                     "frecuencia": fila["frecuencia"], "data": datos}
    mensajes = [
        {"role": "user", "content": fila["pregunta"]},
        _llamada(1, "search_series", {"query": consulta}),
        {"role": "tool", "tool_call_id": "call_1", "name": "search_series", "content": candidatos_json},
        _llamada(2, "get_series", params),
        {"role": "tool", "tool_call_id": "call_2", "name": "get_series",
         "content": json.dumps(respuesta_get, ensure_ascii=False)},
        _llamada(3, "submit_answer", argumentos_submit(fila, datos, params)),
    ]
    return {"qid": fila["qid"], "spec_id": int(fila["spec_id"]), "tipo": fila["tipo"], "estilo": fila["estilo"],
            "serie_id": fila["serie_id"], "familia": fila["familia"], "rango_busqueda": ids.index(fila["serie_id"]) + 1,
            "messages": mensajes}


def construir(preguntas: pd.DataFrame, consultas: dict, buscar) -> tuple[list, Counter]:
    """Trayectorias de todas las preguntas con consulta válida cuya serie aparece en el top 10.

    `buscar(consulta)` devuelve el JSON de candidatos (típicamente `Herramientas.search_series`).
    También devuelve el conteo de preguntas descartadas por motivo.
    """
    trayectorias, perdidas = [], Counter()
    for fila in preguntas.to_dict("records"):
        consulta = consultas.get(fila["qid"])
        if not consulta:
            perdidas["sin consulta generada"] += 1
            continue
        t = armar_trayectoria(fila, consulta, buscar(consulta))
        if t is None:
            perdidas[f"la serie no aparece en el top 10 ({fila['estilo']})"] += 1
            continue
        trayectorias.append(t)
    return trayectorias, perdidas

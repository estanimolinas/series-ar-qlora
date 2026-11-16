"""Corrida de evaluación: preguntas, episodios, guardado incremental con retoma y resumen."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .agente import PROMPT_OPT, episodio
from .evaluador import causa, puntuar, valor_fecha, wilson
from .observabilidad import SinTraza

# corrida -> (prompt, prefijo de archivos, etiqueta del modelo en las trazas, prompt de sistema)
PERFILES = {
    "entrenado": ("basico", "finetuned", "qwen3-4b-instruct-2507-qlora", None),
    "base": ("optimizado", "baseline", "qwen3-4b-instruct-2507-base", PROMPT_OPT),
}
CAMPOS_LOG = ["pasos", "n_search", "n_get", "formato_invalido", "sin_llamada", "contexto_excedido"]


def nombre_corrida(corrida: str, split: str, estilo: str, prueba: bool) -> str:
    prompt, prefijo, _, _ = PERFILES[corrida]
    return f"{prefijo}_{split}_{estilo}_{'prueba' if prueba else 'completa'}_{prompt}"


def cargar_preguntas(datos, split: str, estilo: str, n_prueba: int | None = None) -> pd.DataFrame:
    """Preguntas de evaluación con la respuesta cruda de la API (`data`) de cada especificación."""
    datos = Path(datos)
    f = pd.read_csv(datos / f"eval_{split}_final2.csv")
    f = f[f["marca_var"].fillna("") != "var_sobre_tasa"]          # variaciones sobre tasas: ambiguas, se excluyen
    if estilo == "coloquial":
        f = f[~f["coloquial_ambigua"]]
    s = pd.read_csv(datos / f"specs_{split}_v2.csv")
    s["idx"] = s.index
    f["idx"] = f["spec_id"].str.split("-").str[1].astype(int)
    f = f.merge(s[["idx", "data"]], on="idx", how="left")
    if n_prueba:
        f = f.sample(frac=1, random_state=4).groupby("tipo").head(int(np.ceil(n_prueba / 4)))
    return f.reset_index(drop=True)


def leer_progreso(archivo_local: Path, archivo_drive: Path | None = None) -> set:
    """IDs ya resueltos. Descarta (y reescribe sin ella) una última línea cortada por un corte de sesión."""
    archivo_local = Path(archivo_local)
    if archivo_drive and Path(archivo_drive).exists() and not archivo_local.exists():
        shutil.copy(archivo_drive, archivo_local)
    hechos, validas = set(), []
    if archivo_local.exists():
        for linea in archivo_local.read_text().splitlines(keepends=True):
            try:
                hechos.add(json.loads(linea)["resultado"]["spec_id"])
                validas.append(linea if linea.endswith("\n") else linea + "\n")
            except Exception:
                pass
        archivo_local.write_text("".join(validas))
    return hechos


def ejecutar(preguntas: pd.DataFrame, *, corrida: str, split: str, estilo: str, generar, herramientas,
             archivo_local, guardar=None, cada: int = 25, trazador=None, log=print) -> pd.DataFrame:
    """Evalúa todas las preguntas pendientes y devuelve el DataFrame de resultados (con las ya hechas)."""
    trazador = trazador or SinTraza()
    prompt, _, etiqueta, sistema = PERFILES[corrida]
    archivo_local = Path(archivo_local)
    hechos = leer_progreso(archivo_local)
    log(f"Retomando: ya hechas {len(hechos)} de {len(preguntas)}" if hechos else "Corrida nueva")
    nuevos, t_total, error_score = 0, time.time(), None
    with open(archivo_local, "a") as fout:
        for r in preguntas.to_dict("records"):
            if r["spec_id"] in hechos:
                continue
            params, datos_gold = json.loads(r["params"]), json.loads(r["data"])
            valor, fecha = valor_fecha(r["tipo"], r["frecuencia"], datos_gold, params)
            pregunta = r[estilo]
            t0 = time.time()
            meta = {"modelo": etiqueta, "prompt": prompt, "tipo": r["tipo"], "estilo": estilo,
                    "split": split, "spec_id": r["spec_id"]}
            with trazador.observar("span", "pregunta", input={"pregunta": pregunta}, metadata=meta) as span:
                tid = trazador.id_traza_actual()
                lg, msgs = episodio(pregunta, sistema, generar, herramientas, trazador, etiqueta)
                sc = puntuar(r["tipo"], r["frecuencia"], {"valor": valor, "fecha": fecha},
                             lg["submit"], lg["ids_ultima"], params["ids"])
                span.update(output={"submit": lg["submit"], "correcta": sc["correcta"], "causa": causa(sc)})
                error_score = trazador.puntuar(span, "correcta", sc["correcta"]) or error_score
            fila = {"prompt": prompt, "spec_id": r["spec_id"], "tipo": r["tipo"], "frecuencia": r["frecuencia"],
                    **sc, "causa": causa(sc), **{k: lg[k] for k in CAMPOS_LOG},
                    "segundos": round(time.time() - t0, 1), "trace_id": tid}
            traza = {"prompt": prompt, "spec_id": r["spec_id"], "tipo": r["tipo"], "pregunta": pregunta,
                     "messages": msgs, "log": lg, "score": sc}
            fout.write(json.dumps({"resultado": fila, "traza": traza}, ensure_ascii=False, default=str) + "\n")
            fout.flush()
            nuevos += 1
            if nuevos % cada == 0:
                if guardar:
                    guardar()
                log(f"  [{len(hechos) + nuevos}/{len(preguntas)}] guardado | "
                    f"{(time.time() - t_total) / 60:.1f} min en esta sesión")
    if guardar:
        guardar()
    trazador.flush()
    if error_score:
        log(f"Aviso: los scores no se enviaron a Langfuse ({error_score})")
    return cargar_resultados(archivo_local)[0]


def cargar_resultados(archivo_local):
    """(DataFrame de resultados, lista de trazas) desde el jsonl de una corrida."""
    filas = [json.loads(linea) for linea in Path(archivo_local).read_text().splitlines() if linea.strip()]
    return pd.DataFrame([f["resultado"] for f in filas]), [f["traza"] for f in filas]


def resumen(R: pd.DataFrame, titulo: str = "") -> str:
    """Texto de resumen de una corrida."""
    k, n = int(R["correcta"].sum()), len(R)
    lo, hi = wilson(k, n)
    por_tipo = (R.groupby("tipo")["correcta"].mean() * 100).round(0).to_dict()
    return "\n".join([
        f"### {titulo} | n = {n}",
        f"correcta: {k}/{n} = {k / n:.1%} (IC 95 %: {lo:.0%} a {hi:.0%}) | acepta fracción: "
        f"{R['correcta_flex'].mean():.0%} | terminó: {R['termino'].mean():.0%} | serie correcta: {R['serie_ok'].mean():.0%}",
        f"por tipo: {por_tipo}",
        f"causas: {R['causa'].value_counts().to_dict()}",
        f"pasos (prom.): {R['pasos'].mean():.1f} | JSON inválido: {int(R['formato_invalido'].sum())} | "
        f"sin llamada: {int(R['sin_llamada'].sum())} | contexto excedido: {int(R['contexto_excedido'].sum())} | "
        f"segundos por pregunta: {R['segundos'].mean():.1f}",
    ])

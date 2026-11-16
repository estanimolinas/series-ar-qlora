#!/usr/bin/env python
"""Prepara las preguntas del proyecto en cuatro pasos. Los pasos 2 y 3 llaman a Claude Haiku y cuestan dinero.

    especificaciones   API de datos.gob.ar (gratis): series -> filas con consulta, datos y respuesta final
    preguntas          Haiku: redacta la pregunta coloquial y la técnica; cada una pasa `validar` o se reintenta
    verificar          Haiku: marca las preguntas que cambian el sentido del dato (fidelidad)
    armar              sin costo: une todo, excluye y marca (defectos de período, coloquiales repetidas, etc.)

Costo medido al construir los conjuntos (precio de referencia: USD 0,10 por millón de tokens de entrada y 0,50 de
salida; la fuente de verdad es la consola de Anthropic). Las corridas se retomaron varias veces, así que no hay un
total por conjunto, solo estas corridas:
    preguntas   : 158 especificaciones de train ........ USD 0,089 | 432 de test ........ USD 0,248
    verificar   : train (1.199) USD 0,101 | val (312) USD 0,029 | test (608) USD 0,052
Cada paso guarda el avance cada tantas filas y se puede retomar (solo se rehace lo que falló). Con
ANTHROPIC_API_KEY ausente, los pasos 2 y 3 se niegan a empezar; un error de autenticación corta toda la corrida.

    uv run python scripts/preparar_preguntas.py especificaciones --split test --salida specs_test_nuevo.csv
    uv run python scripts/preparar_preguntas.py preguntas --specs specs_test_nuevo.csv --salida preguntas_test.csv
    uv run python scripts/preparar_preguntas.py verificar --preguntas preguntas_test.csv --salida fidelidad_test.csv
    uv run python scripts/preparar_preguntas.py armar --preguntas preguntas_test.csv \\
        --fidelidad fidelidad_test.csv --salida eval_test_nuevo.csv

Las salidas llevan nombres nuevos: nunca pisan los archivos con los que se entrenó y se evaluó (ver docs/datos_y_licencias.md).
Las especificaciones de val y test se generan con la regla actual (`especificaciones.py`). Las de train se
generaron con una versión anterior y se corrigieron después, así que volver a correr `--split train` produce
especificaciones equivalentes en criterio, no idénticas fila por fila.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from series_qlora import config  # noqa: E402
from series_qlora.cache_api import CacheApi  # noqa: E402
from series_qlora.especificaciones import (  # noqa: E402
    CUPOS,
    especificacion_de_serie_vacia,
    especificaciones_de_serie,
    seleccionar_series,
)
from series_qlora.herramientas import cargar_catalogo  # noqa: E402
from series_qlora.preguntas import (  # noqa: E402
    armar_final,
    costo_usd,
    generar_pregunta,
    leer_fidelidad,
    prompt_fidelidad,
)

MODELO = "claude-haiku-5-5"


def correr(pendientes: list, tarea: Callable, previas: pd.DataFrame, parcial: Path, hilos: int, cada: int) -> pd.DataFrame:
    """Ejecuta `tarea` en paralelo, guarda el avance cada `cada` filas y devuelve previas + nuevas."""
    nuevas, t0 = [], time.time()
    try:
        with ThreadPoolExecutor(max_workers=hilos) as ex:
            for k, salida in enumerate(ex.map(tarea, pendientes), 1):
                nuevas.append(salida)
                if k % cada == 0:
                    pd.concat([previas, pd.DataFrame(nuevas)], ignore_index=True).to_csv(parcial, index=False)
                    print(f"  [{k}/{len(pendientes)}] {time.time() - t0:.0f}s", flush=True)
    finally:
        todas = pd.concat([previas, pd.DataFrame(nuevas)], ignore_index=True) if nuevas else previas
        todas.to_csv(parcial, index=False)
    return todas


def retomar(parcial: Path) -> pd.DataFrame:
    """Filas ya resueltas de una corrida anterior (solo las que salieron bien: las fallidas se rehacen)."""
    if not parcial.exists():
        return pd.DataFrame()
    prev = pd.read_csv(parcial)
    return prev[prev["ok"] == True] if len(prev) else prev  # noqa: E712


def cliente_anthropic():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Falta ANTHROPIC_API_KEY (este paso llama a la API de Anthropic y tiene costo).")
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=3, timeout=90.0), anthropic


def paso_especificaciones(a) -> int:
    d = a.datos
    catalogo = pd.read_csv(d / a.catalogo, low_memory=False)
    con, sin = seleccionar_series(catalogo, a.split, *CUPOS[a.split])
    print(f"{a.split}: series con datos {len(con)} (familias {con['familia'].nunique()}) | "
          f"sin datos {len(sin)} (familias {sin['familia'].nunique()})")
    cache = CacheApi(d / "cache_api.db")
    specs, descartes, t0 = [], Counter(), time.time()
    with ThreadPoolExecutor(max_workers=a.hilos) as ex:
        for k, (o, desc) in enumerate(ex.map(lambda f: especificaciones_de_serie(f, cache.get),
                                             con.to_dict("records")), 1):
            specs += o
            descartes += desc
            if k % 25 == 0:
                print(f"  [{k}/{len(con)}] {time.time() - t0:.0f}s | especificaciones: {len(specs)}", flush=True)
        for o, desc in ex.map(lambda f: especificacion_de_serie_vacia(f, cache.get), sin.to_dict("records")):
            specs += o
            descartes += desc
    if not specs:
        print("No se generó ninguna especificación")
        return 1
    E = pd.DataFrame(specs)
    E["split"] = a.split
    E.to_csv(d / a.salida, index=False)
    print(f"Especificaciones: {len(E)} | series: {E['serie_id'].nunique()} | familias: {E['familia'].nunique()}")
    print(E.groupby(["tipo", "caso"]).size().to_string())
    print("Descartes:", dict(descartes), "| caché:", cache.stats, "| entradas:", cache.tamano())
    cache.cerrar()
    print("Guardado:", d / a.salida)
    return 0


def paso_preguntas(a) -> int:
    d = a.datos
    E = pd.read_csv(d / a.specs)
    E["spec_id"] = E.index                                     # el índice de la tabla es el spec_id
    catalogo = cargar_catalogo(d / a.catalogo)
    cliente, anthropic = cliente_anthropic()
    abortar = threading.Event()

    def llamar(prompt, max_tokens):
        msg = cliente.messages.create(model=MODELO, max_tokens=max_tokens,
                                      messages=[{"role": "user", "content": prompt}])
        txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return txt, msg.stop_reason == "max_tokens", msg.usage.input_tokens, msg.usage.output_tokens

    def tarea(r):
        if abortar.is_set():
            return {"spec_id": r["spec_id"], "ok": False, "error": "abortado"}
        try:
            out = generar_pregunta(r, catalogo.loc[r["serie_id"]], llamar)
        except anthropic.AuthenticationError as e:
            abortar.set()
            return {"spec_id": r["spec_id"], "ok": False, "error": f"AuthenticationError (se corta todo): {e}"[:120]}
        return {**out, "spec_id": r["spec_id"]}

    parcial = d / (Path(a.salida).stem + "_parcial.csv")
    previas = retomar(parcial)
    hechas = set(previas["spec_id"]) if len(previas) else set()
    orden = E.sample(frac=1, random_state=1)
    pendientes = [r for r in orden.to_dict("records") if r["spec_id"] not in hechas]
    print(f"Especificaciones: {len(E)} | ya hechas: {len(hechas)} | pendientes: {len(pendientes)}")
    t0 = time.time()
    todas = correr(pendientes, tarea, previas, parcial, a.hilos, cada=50)
    R = todas.iloc[len(previas):] if len(previas) else todas
    if len(R):
        print(f"Esta corrida: {int(R['ok'].sum())}/{len(R)} ok | costo ${costo_usd(R['ent'].fillna(0).sum(), R['sal'].fillna(0).sum()):.4f} "
              f"| {time.time() - t0:.0f}s | fallos: {Counter(R[~R['ok']]['error']).most_common(3)}")
    if abortar.is_set():
        print("*** Se cortó por error de autenticación: revisá la clave y volvé a ejecutar ***")
        return 1
    ok = todas[todas["ok"] == True].merge(  # noqa: E712
        E[["spec_id", "tipo", "caso", "serie_id", "familia", "frecuencia", "params"]], on="spec_id")
    ok.drop(columns=["ok", "error", "ent", "sal"]).to_csv(d / a.salida, index=False)
    print(f"Preguntas válidas: {len(ok)} de {len(E)} | guardado: {d / a.salida}")
    return 0


def paso_verificar(a) -> int:
    d = a.datos
    Q = pd.read_csv(d / a.preguntas)
    catalogo = cargar_catalogo(d / a.catalogo)
    cliente, anthropic = cliente_anthropic()
    abortar = threading.Event()

    def tarea(r):
        if abortar.is_set():
            return {"spec_id": r["spec_id"], "ok": False, "error": "abortado"}
        try:
            msg = cliente.messages.create(
                model=MODELO, max_tokens=500, extra_body={"thinking": {"type": "disabled"}},
                messages=[{"role": "user", "content": prompt_fidelidad(r, catalogo.loc[r["serie_id"]])}])
            txt = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            return {"spec_id": r["spec_id"], "ok": True, "error": "", **leer_fidelidad(txt),
                    "ent": msg.usage.input_tokens, "sal": msg.usage.output_tokens}
        except anthropic.AuthenticationError as e:
            abortar.set()
            return {"spec_id": r["spec_id"], "ok": False, "error": f"AuthenticationError (se corta todo): {e}"[:120]}
        except Exception as e:
            return {"spec_id": r["spec_id"], "ok": False, "error": f"{type(e).__name__}: {str(e)[:100]}"}

    parcial = d / (Path(a.salida).stem + "_parcial.csv")
    previas = retomar(parcial)
    hechas = set(previas["spec_id"]) if len(previas) else set()
    pendientes = [r for r in Q.to_dict("records") if r["spec_id"] not in hechas]
    print(f"Preguntas: {len(Q)} | ya verificadas: {len(hechas)} | pendientes: {len(pendientes)}")
    t0 = time.time()
    todas = correr(pendientes, tarea, previas, parcial, a.hilos, cada=100)
    R = todas.iloc[len(previas):] if len(previas) else todas
    if len(R):
        print(f"Esta corrida: {int(R['ok'].sum())}/{len(R)} ok | costo ${costo_usd(R['ent'].fillna(0).sum(), R['sal'].fillna(0).sum()):.4f} "
              f"| {time.time() - t0:.0f}s | fallos: {Counter(R[~R['ok']]['error']).most_common(3)}")
    if abortar.is_set():
        print("*** Se cortó por error de autenticación: revisá la clave y volvé a ejecutar ***")
        return 1
    V = todas[todas["ok"] == True]  # noqa: E712
    V.to_csv(d / a.salida, index=False)
    print(f"Verificadas: {len(V)} de {len(Q)} | marcadas: coloquial {(~V['col_ok'].astype(bool)).mean():.1%}, "
          f"técnica {(~V['tec_ok'].astype(bool)).mean():.1%} | guardado: {d / a.salida}")
    return 0


def paso_armar(a) -> int:
    d = a.datos
    Q, V = pd.read_csv(d / a.preguntas), pd.read_csv(d / a.fidelidad)
    V = V.drop_duplicates("spec_id", keep="last")
    finales, excluidas = armar_final(Q, V, cargar_catalogo(d / a.catalogo), exigir_verificacion=a.exigir)
    print(f"Preguntas: {len(Q)} | finales: {len(finales)} | excluidas: {len(excluidas)}")
    print("Por tipo:", finales["tipo"].value_counts().to_dict(), "| % sin_datos:",
          f"{(finales['tipo'] == 'sin_datos').mean():.1%}")
    print("Coloquiales ambiguas:", int(finales["coloquial_ambigua"].sum()), "| marca_var:",
          finales["marca_var"].value_counts().to_dict())
    finales.to_csv(d / a.salida, index=False)
    excluidas.to_csv(d / (Path(a.salida).stem + "_excluidas.csv"), index=False)
    print("Guardado:", d / a.salida)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=config.directorio_datos())
    ap.add_argument("--catalogo", default="catalogo_busqueda.csv")
    ap.add_argument("--hilos", type=int, default=4)
    sub = ap.add_subparsers(dest="paso", required=True)
    s = sub.add_parser("especificaciones")
    s.add_argument("--split", choices=list(CUPOS), required=True)
    s.add_argument("--salida", required=True)
    s = sub.add_parser("preguntas")
    s.add_argument("--specs", required=True)
    s.add_argument("--salida", required=True)
    s = sub.add_parser("verificar")
    s.add_argument("--preguntas", required=True)
    s.add_argument("--salida", required=True)
    s = sub.add_parser("armar")
    s.add_argument("--preguntas", required=True)
    s.add_argument("--fidelidad", required=True)
    s.add_argument("--salida", required=True)
    s.add_argument("--exigir", action="store_true", help="falla si alguna pregunta no fue verificada")
    a = ap.parse_args(argv)
    return {"especificaciones": paso_especificaciones, "preguntas": paso_preguntas,
            "verificar": paso_verificar, "armar": paso_armar}[a.paso](a)


if __name__ == "__main__":
    sys.exit(main())

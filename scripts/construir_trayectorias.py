#!/usr/bin/env python
"""Arma las trayectorias de entrenamiento (pregunta -> search_series -> get_series -> submit_answer).

La consulta de búsqueda de cada pregunta la escribe Claude Haiku (unos USD 0,09 por 1.900 preguntas) y se guarda
en consultas_busqueda_train_parcial.csv; lo demás sale de los datos guardados. Con `--solo-armar` no se llama a
la API: se usan las consultas ya guardadas, que es como se reproduce el conjunto sin gastar nada.

    uv run python scripts/construir_trayectorias.py --datos data --solo-armar --comparar trayectorias_train_v1.jsonl

Catálogo: las trayectorias se armaron con el índice completo (catalogo_busqueda.csv, 8.226 series). Con el
catálogo anterior al dataset de tipo de cambio (8.209 series) 6 de las 1.705 trayectorias cambian: el índice tiene
17 documentos menos, bm25 reordena empates en los últimos puestos del top 10 y una serie de ese dataset ya no
aparece. La partición, en cambio, se publicó sobre el catálogo anterior (ver scripts/particionar.py).

Entradas en --datos: specs_train_v1.csv, train_final2.csv, el catálogo y consultas_busqueda_train_parcial.csv.
Salida: trayectorias_train.jsonl (no pisa trayectorias_train_v1.jsonl, que es el que se usó para entrenar).
Para llamar a la API: variable de entorno ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from series_qlora import config  # noqa: E402
from series_qlora.esquema import HERRAMIENTAS  # noqa: E402
from series_qlora.herramientas import (  # noqa: E402
    Herramientas,
    cargar_catalogo,
    construir_indice,
    crear_cache,
)
from series_qlora.trayectorias import (  # noqa: E402
    construir,
    costo_usd,
    pedir_consulta,
    preguntas_de_entrenamiento,
)


def generar_consultas(preguntas, catalogo, previas: pd.DataFrame, parcial: Path, hilos: int) -> pd.DataFrame:
    """Pide a Claude las consultas que faltan; guarda el avance cada 200 y devuelve todas (previas y nuevas)."""
    import anthropic
    cliente = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=3, timeout=60.0)
    hechas = set(previas["qid"])
    pendientes = [r for r in preguntas.to_dict("records") if r["qid"] not in hechas]
    print(f"Consultas ya hechas: {len(hechas)} | pendientes: {len(pendientes)}")
    if not pendientes:
        return previas
    nuevas, t0 = [], time.time()

    def tarea(r):
        return pedir_consulta(cliente, r, catalogo, errores_fatales=(anthropic.AuthenticationError,))

    try:
        with ThreadPoolExecutor(max_workers=hilos) as ex:
            for k, salida in enumerate(ex.map(tarea, pendientes), 1):
                nuevas.append(salida)
                if k % 200 == 0:
                    pd.concat([previas, pd.DataFrame(nuevas)], ignore_index=True).to_csv(parcial, index=False)
                    print(f"  [{k}/{len(pendientes)}] {time.time() - t0:.0f}s", flush=True)
    finally:
        todas = pd.concat([previas, pd.DataFrame(nuevas)], ignore_index=True)
        todas.to_csv(parcial, index=False)
    R = pd.DataFrame(nuevas)
    print(f"Esta corrida: {int(R['ok'].sum())}/{len(R)} ok | costo ${costo_usd(R):.4f} | {time.time() - t0:.0f}s")
    print("Fallos:", Counter(R[~R['ok']]['error']).most_common(3))
    return todas


def comparar(trayectorias: list, referencia: Path) -> bool:
    ref = [json.loads(linea) for linea in referencia.read_text(encoding="utf-8").splitlines() if linea.strip()]
    nuevas = {t["qid"]: t for t in trayectorias}
    viejas = {t["qid"]: t for t in ref}
    solo_nuevas, solo_viejas = set(nuevas) - set(viejas), set(viejas) - set(nuevas)
    distintas = [q for q in nuevas.keys() & viejas.keys() if nuevas[q] != viejas[q]]
    print(f"Comparación con {referencia.name}: {len(ref)} de referencia | {len(trayectorias)} nuevas | "
          f"solo nuevas: {len(solo_nuevas)} | solo en la referencia: {len(solo_viejas)} | con contenido distinto: {len(distintas)}")
    return not (solo_nuevas or solo_viejas or distintas)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=config.directorio_datos())
    ap.add_argument("--catalogo", default="catalogo_busqueda.csv")
    ap.add_argument("--solo-armar", action="store_true", help="no llamar a la API: usar las consultas ya guardadas")
    ap.add_argument("--n-prueba", type=int, default=None)
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--comparar", default=None, help="archivo de trayectorias contra el cual comparar el resultado")
    ap.add_argument("--salida", default="trayectorias_train.jsonl")
    a = ap.parse_args(argv)

    d = a.datos
    preguntas = preguntas_de_entrenamiento(pd.read_csv(d / "specs_train_v1.csv"), pd.read_csv(d / "train_final2.csv"))
    print(f"Preguntas candidatas: {len(preguntas)} | por tipo: {preguntas['tipo'].value_counts().to_dict()}")
    if a.n_prueba:
        preguntas = preguntas.head(a.n_prueba)
    catalogo = cargar_catalogo(d / a.catalogo)

    parcial = d / "consultas_busqueda_train_parcial.csv"
    previas = pd.read_csv(parcial) if parcial.exists() else pd.DataFrame(columns=["qid", "ok", "consulta"])
    previas = previas[previas["ok"] == True]  # noqa: E712
    consultas_df = previas if a.solo_armar else generar_consultas(preguntas, catalogo, previas, parcial, a.hilos)
    buenas = consultas_df[consultas_df["ok"] == True]  # noqa: E712
    consultas = dict(zip(buenas["qid"], buenas["consulta"], strict=True))

    with tempfile.TemporaryDirectory() as tmp:
        crear_cache(Path(tmp) / "cache.db")
        construir_indice(Path(tmp) / "indice.db", catalogo)
        h = Herramientas(Path(tmp) / "cache.db", Path(tmp) / "indice.db", catalogo)
        trayectorias, perdidas = construir(preguntas, consultas, h.search_series)
        h.cerrar()

    print(f"\nTrayectorias: {len(trayectorias)} de {len(preguntas)} | perdidas: {dict(perdidas)}")
    if trayectorias:
        T = pd.DataFrame([{k: v for k, v in t.items() if k != "messages"} for t in trayectorias])
        print("Por tipo:", T["tipo"].value_counts().to_dict(), "| por estilo:", T["estilo"].value_counts().to_dict())
        print(f"% sin_datos: {(T['tipo'] == 'sin_datos').mean():.1%} | familias: {T['familia'].nunique()} | "
              f"series: {T['serie_id'].nunique()} | rango del buscador (mediana, máx): "
              f"{int(T['rango_busqueda'].median())}, {int(T['rango_busqueda'].max())}")
    salida = d / a.salida
    salida.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trayectorias), encoding="utf-8")
    print("Guardado:", salida)

    ok = True
    if a.comparar:
        ok = comparar(trayectorias, d / a.comparar)
        herr = d / "tools_v1.json"
        if herr.exists():
            igual = json.loads(herr.read_text()) == HERRAMIENTAS
            print("Esquema de herramientas igual al de tools_v1.json:", igual)
            ok = ok and igual
        print("Coincide con la referencia" if ok else "NO coincide con la referencia")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

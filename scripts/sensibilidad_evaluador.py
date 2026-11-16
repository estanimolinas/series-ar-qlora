#!/usr/bin/env python
"""Análisis de sensibilidad: ¿cuánto cambia el resultado si la serie cuenta como correcta con cualquier consulta?

    uv run python scripts/sensibilidad_evaluador.py --datos data

La métrica principal mira solo la última consulta a get_series. Aquí se repuntúan los episodios ya guardados de test
(sin volver a correr ningún modelo) con la variante `puntuar_alguna_serie` y se compara con la métrica principal. Es un
análisis de robustez: la métrica principal no cambia.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from series_qlora import config  # noqa: E402
from series_qlora.evaluador import ids_consultados, puntuar_alguna_serie, wilson  # noqa: E402

CORRIDAS = {"Base + prompt optimizado": "baseline_test_tecnica_completa_optimizado",
            "QLoRA, sin prompt de sistema": "finetuned_test_tecnica_completa_basico"}


def repuntuar(datos: Path, prefijo: str, objetivos: dict) -> pd.DataFrame:
    """Resultados de la corrida con las columnas nuevas y la cantidad de series consultadas."""
    res = pd.read_csv(datos / f"{prefijo}.csv").set_index("spec_id")
    filas = []
    for linea in (datos / f"{prefijo}_episodios.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(linea)
        ids = ids_consultados(e["messages"])
        extra = puntuar_alguna_serie(res.loc[e["spec_id"]].to_dict(), ids, objetivos[e["spec_id"]])
        filas.append({"spec_id": e["spec_id"], "n_series_consultadas": len(set(ids)), **extra})
    return res.join(pd.DataFrame(filas).set_index("spec_id"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=config.directorio_datos())
    a = ap.parse_args(argv)
    objetivos = pd.read_csv(a.datos / "eval_test_final2.csv").set_index("spec_id")["serie_id"].to_dict()
    for nombre, prefijo in CORRIDAS.items():
        d = repuntuar(a.datos, prefijo, objetivos)
        n = len(d)
        k1, k2 = int(d["correcta"].sum()), int(d["correcta_alguna"].sum())
        multi = d[d["n_series_consultadas"] > 1]
        print(f"\n{nombre} (n = {n})")
        print(f"  métrica principal (última consulta): {k1}/{n} = {k1 / n:.1%} (IC 95 % de Wilson: "
              f"{wilson(k1, n)[0]:.1%} a {wilson(k1, n)[1]:.1%})")
        print(f"  variante (cualquier consulta)      : {k2}/{n} = {k2 / n:.1%} (IC 95 % de Wilson: "
              f"{wilson(k2, n)[0]:.1%} a {wilson(k2, n)[1]:.1%})")
        print(f"  episodios con más de una serie consultada: {len(multi)} | recuperados por la variante: {k2 - k1}")
        if k2 < k1:
            print("  AVISO: la variante da menos aciertos que la principal; revisar")
    return 0


if __name__ == "__main__":
    sys.exit(main())

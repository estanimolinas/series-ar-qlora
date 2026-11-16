#!/usr/bin/env python
"""Genera las dos figuras animadas (SVG) a partir de los resultados de test, sin GPU ni API.

    uv run python scripts/figuras_animadas.py --datos data --destino figures

Usan los mismos datos que las figuras estáticas del notebook 09 (los CSV de test de las dos corridas) y los valores del
registro de entrenamiento. Las estáticas (PNG y PDF) son las que se citan; estas son una versión para el README.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from series_qlora import config  # noqa: E402
from series_qlora.graficos_animados import generar_animadas  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=config.directorio_datos())
    ap.add_argument("--destino", default="figures")
    a = ap.parse_args(argv)
    A = pd.read_csv(a.datos / "baseline_test_tecnica_completa_optimizado.csv")
    B = pd.read_csv(a.datos / "finetuned_test_tecnica_completa_basico.csv")
    if len(A) != len(B):
        print(f"Las dos corridas no tienen las mismas preguntas ({len(A)} y {len(B)})")
        return 1
    for ruta in generar_animadas(A, B, a.destino):
        print("Guardado:", ruta)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Reparte los datasets del catálogo en train/val/test por familias.

    python scripts/particionar.py --datos <carpeta> --catalogo catalogo_busqueda_antes_tc.csv --verificar
    python scripts/particionar.py --datos <carpeta> --guardar

`--verificar` compara con la partición publicada (puntaje, semilla y series usables por split) y termina con
error si difiere. Esa partición se calculó sobre el catálogo de 8.209 series, antes de agregar el dataset de tipo
de cambio (que quedó en train, en una familia propia): sobre un catálogo con más datasets o temas, el reparto
cambia, por eso la verificación usa el catálogo anterior. `--guardar` escribe particion_datasets.csv
(dataset_id, familia, split) y no pisa un archivo existente salvo con `--forzar`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from series_qlora import config  # noqa: E402
from series_qlora.particion import particionar  # noqa: E402

PUBLICADA = {"puntaje": 0.689, "semilla": 3479, "usables": {"train": 5696, "val": 417, "test": 408}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=config.directorio_datos())
    ap.add_argument("--catalogo", default="catalogo_busqueda.csv", help="archivo del catálogo dentro de --datos")
    ap.add_argument("--verificar", action="store_true", help="comparar con la partición publicada")
    ap.add_argument("--guardar", action="store_true")
    ap.add_argument("--forzar", action="store_true", help="permitir sobrescribir particion_datasets.csv")
    a = ap.parse_args(argv)

    df = pd.read_csv(a.datos / a.catalogo, low_memory=False)
    df = df.drop(columns=[c for c in ("familia", "split") if c in df.columns])
    r = particionar(df)
    print(f"Puntaje {r['puntaje']:.3f} (semilla {r['semilla']}) | series usables: {r['usables']}")

    if a.verificar:
        coincide = (abs(r["puntaje"] - PUBLICADA["puntaje"]) < 0.002 and r["semilla"] == PUBLICADA["semilla"]
                    and r["usables"] == PUBLICADA["usables"])
        print("Coincide con la partición publicada" if coincide else "NO coincide con la partición publicada")
        if not coincide:
            return 1
    if a.guardar:
        destino = a.datos / "particion_datasets.csv"
        if destino.exists() and not a.forzar:
            print(f"{destino} ya existe: no se sobrescribe sin --forzar (la partición publicada está congelada)")
            return 1
        r["df"][["dataset_id", "familia", "split"]].drop_duplicates().to_csv(destino, index=False)
        print("Guardado:", destino)
    return 0


if __name__ == "__main__":
    sys.exit(main())

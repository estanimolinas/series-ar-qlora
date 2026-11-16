#!/usr/bin/env python
"""Entrena el adaptador QLoRA (Qwen3-4B-Instruct en 4 bits) sobre las trayectorias de train.

    uv run python scripts/entrenar.py --datos data --modo datos        # sin GPU: prepara y verifica la separación
    python scripts/entrenar.py --datos <carpeta> --modo completo       # Colab, GPU T4: unos 75 minutos

Modos: `datos` (solo cuenta ejemplos de entrenamiento y de control), `prueba_10_pasos` y `completo`. Entrada:
trayectorias_train_v1.jsonl y tools_v1.json en --datos. Salida: <datos>/qlora_<modo>/ (checkpoints y adapter/).
Si se corta, volver a ejecutar el mismo comando retoma desde el último checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from series_qlora import config, entrenamiento  # noqa: E402

PUBLICADO = {"entrenamiento": 1586, "control": 119}      # entrenamiento v1: 100 pasos de 16 ejemplos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=config.directorio_datos())
    ap.add_argument("--modo", choices=["datos", "prueba_10_pasos", "completo"], default="datos")
    ap.add_argument("--trayectorias", default="trayectorias_train_v1.jsonl")
    ap.add_argument("--epocas", type=int, default=1)
    a = ap.parse_args(argv)

    trayectorias = entrenamiento.cargar_trayectorias(a.datos / a.trayectorias)
    train, control = entrenamiento.separar(trayectorias)
    print(f"Trayectorias: {len(trayectorias)} | entrenamiento: {len(train)} | control: {len(control)} | "
          f"pasos por época: {entrenamiento.pasos_por_epoca(len(train))}")
    if a.modo == "datos":
        coincide = {"entrenamiento": len(train), "control": len(control)} == PUBLICADO
        print("Coincide con el entrenamiento publicado" if coincide else "NO coincide con el entrenamiento publicado")
        return 0 if coincide else 1

    herramientas = json.loads((a.datos / "tools_v1.json").read_text(encoding="utf-8"))
    entrenamiento.entrenar(trayectorias, herramientas, a.datos / f"qlora_{a.modo}", a.modo, a.epocas)
    return 0


if __name__ == "__main__":
    sys.exit(main())

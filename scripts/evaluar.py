#!/usr/bin/env python
"""Evalúa el agente sobre val o test: modelo base (prompt optimizado) o adaptador QLoRA (sin prompt).

Uso en Colab (T4), con los datos en Drive:
    export SERIES_QLORA_DATA=/content/drive/MyDrive/qlora-series-ar/data
    python scripts/evaluar.py --corrida entrenado --split test --langfuse

Una corrida por sesión. Si se corta, se vuelve a ejecutar el mismo comando: retoma donde quedó.
Regla para test: no se toca nada (prompt, agente, adaptador) después de ver el resultado.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from series_qlora import config, corrida  # noqa: E402
from series_qlora.herramientas import Herramientas, cargar_catalogo  # noqa: E402
from series_qlora.observabilidad import SinTraza, TrazadorLangfuse  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrida", choices=["entrenado", "base"], required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--estilo", choices=["tecnica", "coloquial"], default="tecnica")
    ap.add_argument("--n-prueba", type=int, default=None, help="prueba de humo con N preguntas (solo val)")
    ap.add_argument("--datos", type=Path, default=config.directorio_datos())
    ap.add_argument("--adaptador", type=Path, default=None, help="por defecto: <datos>/qlora_completo/adapter")
    ap.add_argument("--trabajo", type=Path, default=Path("/tmp/series_qlora_trabajo"),
                    help="copia local de las bases SQLite (SQLite sobre Drive es lento y frágil)")
    ap.add_argument("--langfuse", action="store_true",
                    help="enviar trazas y scores (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)")
    a = ap.parse_args(argv)
    if a.split == "test" and a.n_prueba:
        ap.error("no se hacen pruebas de humo sobre test")

    datos, trabajo = a.datos, a.trabajo
    trabajo.mkdir(parents=True, exist_ok=True)
    adaptador = a.adaptador or datos / "qlora_completo" / "adapter"
    nombre = corrida.nombre_corrida(a.corrida, a.split, a.estilo, bool(a.n_prueba))
    arch_drive, arch_local = datos / f"{nombre}_resultados.jsonl", trabajo / f"{nombre}_resultados.jsonl"
    if arch_drive.exists() and not arch_local.exists():
        shutil.copy(arch_drive, arch_local)

    for db in ("cache_api.db", "busqueda.db"):
        if not (trabajo / db).exists():
            shutil.copy(datos / db, trabajo / db)

    from series_qlora import modelo
    herramientas_json = json.loads((datos / "tools_v1.json").read_text())
    mdl, tok = modelo.cargar(a.corrida, str(adaptador))
    generar = modelo.crear_generador(mdl, tok, herramientas_json)

    trazador = SinTraza()
    if a.langfuse:
        from langfuse import get_client
        lf = get_client()
        if not lf.auth_check():
            sys.exit("Langfuse: auth_check falló (revisar claves y región de LANGFUSE_HOST)")
        trazador = TrazadorLangfuse(lf)

    catalogo = cargar_catalogo(datos / "catalogo_busqueda.csv")
    herr = Herramientas(trabajo / "cache_api.db", trabajo / "busqueda.db", catalogo)

    def guardar():
        herr.cache.commit()
        shutil.copy(arch_local, arch_drive)
        shutil.copy(trabajo / "cache_api.db", datos / "cache_api.db")

    preguntas = corrida.cargar_preguntas(datos, a.split, a.estilo, a.n_prueba)
    print(f"Corrida: {a.corrida} | split: {a.split} | estilo: {a.estilo} | preguntas: {len(preguntas)} | "
          f"por tipo: {preguntas['tipo'].value_counts().to_dict()}")
    R = corrida.ejecutar(preguntas, corrida=a.corrida, split=a.split, estilo=a.estilo, generar=generar,
                         herramientas=herr, archivo_local=arch_local, guardar=guardar,
                         cada=4 if a.n_prueba else 25, trazador=trazador)
    _, trazas = corrida.cargar_resultados(arch_local)
    R.to_csv(datos / f"{nombre}.csv", index=False)
    with open(datos / f"{nombre}_episodios.jsonl", "w") as f:
        for t in trazas:
            f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
    print(corrida.resumen(R, f"{a.corrida} | {a.split}"))
    print(f"Guardado: {nombre}.csv y {nombre}_episodios.jsonl en {datos}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

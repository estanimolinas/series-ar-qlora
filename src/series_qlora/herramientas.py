"""Herramientas del agente: search_series (SQLite FTS5) y get_series (API con caché en SQLite).

El agente nunca llama a la API directamente: get_series busca primero en una caché cuya clave son los
parámetros normalizados. Así las corridas son reproducibles y no dependen de que la API siga respondiendo igual.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .config import API_BASE, HEADERS, PARAMS_TOOL
from .texto import STOP, norm

ESQUEMA_CACHE = "CREATE TABLE IF NOT EXISTS cache (clave TEXT PRIMARY KEY, status INTEGER, cuerpo TEXT, fecha TEXT)"


def clave(params: dict) -> str:
    """Clave canónica de un pedido: parámetros no nulos, como texto, ordenados por nombre."""
    p = {k: str(v) for k, v in params.items() if v is not None}
    return json.dumps(dict(sorted(p.items())), ensure_ascii=False)


def cargar_catalogo(ruta) -> pd.DataFrame:
    """Catálogo de series indexado por serie_id (sin duplicados)."""
    return pd.read_csv(ruta, low_memory=False).drop_duplicates("serie_id").set_index("serie_id")


def crear_cache(ruta) -> None:
    con = sqlite3.connect(ruta)
    con.execute(ESQUEMA_CACHE)
    con.commit()
    con.close()


def textos_indice(catalogo: pd.DataFrame) -> pd.DataFrame:
    """Textos que se indexan: `titulo` (título de la serie) y `contexto` (todo lo demás que la describe).

    El contexto reúne distribución, descripción, dataset, tema, unidades y frecuencia: así una consulta
    como "mensual" o "precios" encuentra series cuyo título no las nombra.
    """
    frecuencia = catalogo["frecuencia"].fillna("") if "frecuencia" in catalogo else ""
    contexto = (catalogo["distribucion_titulo"].map(norm) + " " + catalogo["serie_descripcion"].map(norm)
                + " " + catalogo["dataset_titulo"].map(norm) + " " + catalogo["dataset_tema"].map(norm)
                + " " + catalogo["serie_unidades"].map(norm) + " " + frecuencia)
    return pd.DataFrame({"titulo": catalogo["serie_titulo"].map(norm), "contexto": contexto})


def construir_indice(ruta, catalogo: pd.DataFrame) -> None:
    """Índice FTS5 con columnas serie_id (sin indexar), titulo y contexto.

    El ranking usa bm25(fts, 0.0, 2.0, 1.0): el título pesa el doble que el contexto.
    """
    Path(ruta).unlink(missing_ok=True)
    con = sqlite3.connect(ruta)
    con.execute("CREATE VIRTUAL TABLE fts USING fts5("
                "serie_id UNINDEXED, titulo, contexto, tokenize='unicode61 remove_diacritics 2')")
    t = textos_indice(catalogo)
    con.executemany("INSERT INTO fts(serie_id, titulo, contexto) VALUES (?,?,?)",
                    zip(t.index, t["titulo"], t["contexto"], strict=True))
    con.commit()
    con.close()


def consultar_api(params: dict, api_base: str = API_BASE, intentos: int = 3,
                  get: Callable | None = None, dormir: Callable[[float], None] = time.sleep):
    """GET con reintentos. Devuelve (status, json); (None, None) si no hubo respuesta usable.

    Se aceptan 200 y 400 (un 400 es información: la API rechazó esos parámetros y queda en la caché).
    """
    if get is None:
        import requests
        get = requests.get
    for n in range(intentos):
        try:
            r = get(api_base, params=params, headers=HEADERS, timeout=30)
            if r.status_code in (200, 400):
                return r.status_code, r.json()
        except Exception:
            pass
        dormir(1 + n)
    return None, None


class Herramientas:
    """search_series y get_series. `api` es inyectable para pruebas: callable(params) -> (status, json)."""

    def __init__(self, ruta_cache, ruta_indice, catalogo: pd.DataFrame,
                 api: Callable[[dict], tuple] | None = None):
        self.cache = sqlite3.connect(ruta_cache)
        self.cache.execute(ESQUEMA_CACHE)
        self.fts = sqlite3.connect(ruta_indice)
        self.catalogo = catalogo
        self.api = api or consultar_api
        self.fallos_cache = 0     # pedidos que no estaban en la caché
        self.sin_red = 0          # pedidos que no se pudieron resolver

    # ---- search_series ----
    def search_series(self, query: str, k: int = 10) -> str:
        toks = [t for t in re.findall(r"[a-z0-9]+", norm(query)) if t not in STOP and len(t) > 1]
        toks = list(dict.fromkeys(toks))
        if not toks:
            return "[]"
        # Palabras largas: solo prefijo de 5 caracteres (tolera plurales y derivaciones)
        partes = [f'"{t[:5]}"*' if len(t) >= 6 else f'"{t}"' for t in toks]
        # El mismo puntaje puede diferir en el último bit según la compilación de SQLite (arm64 vs x86): se redondea
        # y se desempata por fila del catálogo para que el top-k no dependa de la máquina
        filas = self.fts.execute(
            "SELECT serie_id, bm25(fts, 0.0, 2.0, 1.0) AS s FROM fts WHERE fts MATCH ? "
            "ORDER BY round(s, 9), rowid LIMIT ?",
            (" OR ".join(partes), k)).fetchall()
        cat = self.catalogo
        return json.dumps([{"serie_id": s, "titulo": str(cat.loc[s, "serie_titulo"]),
                            "distribucion": str(cat.loc[s, "distribucion_titulo"]),
                            "frecuencia": str(cat.loc[s, "frecuencia"])} for s, _ in filas],
                          ensure_ascii=False)

    # ---- get_series ----
    def get_series(self, args: dict) -> str:
        extra = set(args) - PARAMS_TOOL
        if extra or "ids" not in args:
            return json.dumps({"status": 400,
                               "error": f"Parámetros no permitidos o faltantes: {sorted(extra) or ['ids']}"},
                              ensure_ascii=False)
        k = clave(args)
        fila = self.cache.execute("SELECT status, cuerpo FROM cache WHERE clave=?", (k,)).fetchone()
        if fila:
            status, js = fila[0], json.loads(fila[1])
        else:
            self.fallos_cache += 1
            status, js = self.api(json.loads(k))
            if status is None:
                self.sin_red += 1
                return json.dumps({"status": None, "error": "no se pudo consultar la API"}, ensure_ascii=False)
            self.cache.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
                (k, status, json.dumps(js, ensure_ascii=False),
                 time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())))
            self.cache.commit()
        if status != 200:
            return json.dumps({"status": status, "error": js.get("errors", js)}, ensure_ascii=False)
        sid = str(args["ids"])
        c = self.catalogo.loc[sid] if sid in self.catalogo.index else None
        return json.dumps({
            "status": 200, "serie_id": sid,
            "titulo": None if c is None else str(c["serie_titulo"]),
            "unidades": None if (c is None or pd.isna(c["serie_unidades"])) else c["serie_unidades"],
            "frecuencia": None if c is None else str(c["frecuencia"]),
            "data": js.get("data", [])}, ensure_ascii=False)

    def cerrar(self) -> None:
        self.cache.commit()
        self.cache.close()
        self.fts.close()

"""Caché de pedidos a la API de series de tiempo, segura para varios hilos.

Es la caché con la que se construyeron las especificaciones (`cache_api.db`): clave = parámetros normalizados,
solo se agrega (nunca se pisa una respuesta guardada, salvo que la API se vuelva a consultar a propósito).
Su esquema (`consultado`) es el de ese archivo; la caché que usa el agente (`herramientas.py`) usa `fecha`.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

from .herramientas import clave as clave_canonica
from .herramientas import consultar_api

PARAMS_API = frozenset({"ids", "start_date", "end_date", "limit", "start", "sort",
                        "collapse", "collapse_aggregation", "representation_mode"})
ESQUEMA = ("CREATE TABLE IF NOT EXISTS cache (clave TEXT PRIMARY KEY, status INTEGER, "
           "cuerpo TEXT, consultado TEXT)")


def clave(params: dict) -> str:
    """Clave canónica; rechaza parámetros que la API no define, para que un error de tipeo no quede en la caché."""
    extra = set(params) - PARAMS_API
    if extra:
        raise ValueError(f"Parámetros no previstos: {sorted(extra)}")
    return clave_canonica(params)


class CacheApi:
    """get(params) -> {'status', 'json', 'consultado', 'de_cache'}; status None si la red falló.

    `api` es inyectable (callable(params) -> (status, json)) para probar sin red; `pausa` espacia las consultas
    reales para no saturar la API.
    """

    def __init__(self, ruta, api: Callable[[dict], tuple] | None = None, pausa: float = 0.3,
                 dormir: Callable[[float], None] = time.sleep):
        self.con = sqlite3.connect(ruta, check_same_thread=False)
        self.con.execute(ESQUEMA)
        self.con.commit()
        self.lock = threading.Lock()
        self.api, self.pausa, self.dormir = api or consultar_api, pausa, dormir
        self.stats = {"aciertos": 0, "fallos_de_cache": 0, "error_de_red": 0}

    def get(self, params: dict) -> dict:
        k = clave(params)
        with self.lock:
            fila = self.con.execute("SELECT status, cuerpo, consultado FROM cache WHERE clave=?", (k,)).fetchone()
            self.stats["aciertos" if fila else "fallos_de_cache"] += 1
        if fila:
            return {"status": fila[0], "json": json.loads(fila[1]), "consultado": fila[2], "de_cache": True}
        status, js = self.api(json.loads(k))
        self.dormir(self.pausa)
        if status is None:
            with self.lock:
                self.stats["error_de_red"] += 1
            return {"status": None, "json": None, "consultado": None, "de_cache": False}
        ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.lock:
            self.con.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
                             (k, status, json.dumps(js, ensure_ascii=False), ahora))
            self.con.commit()
        return {"status": status, "json": js, "consultado": ahora, "de_cache": False}

    def tamano(self) -> int:
        with self.lock:
            return self.con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    def cerrar(self) -> None:
        with self.lock:
            self.con.commit()
            self.con.close()

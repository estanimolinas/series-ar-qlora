"""Especificaciones: de una serie del catálogo a filas {pregunta posible, consulta a la API, respuesta final}.

Cada especificación sale del código y de la respuesta real de la API (no de un modelo): el tipo de pregunta, los
parámetros de la consulta, los datos devueltos y la respuesta final esperada. Las preguntas en lenguaje natural se
escriben después (ver `preguntas.py`) a partir de estas filas.

Tipos: ultimo_dato, valor_puntual, variacion_interanual y sin_datos (período fuera del rango de la serie).
`get` es cualquier callable(params) -> {'status', 'json', 'consultado'} (normalmente `CacheApi.get`).
"""
from __future__ import annotations

import json
import random
import zlib
from collections import Counter
from collections.abc import Callable

import pandas as pd

from .evaluador import periodo

LAG = {"mensual": 12, "trimestral": 4, "anual": 1}        # diarias y semestrales quedan fuera de esta versión
PASO = {"mensual": 1, "trimestral": 3, "anual": 12}       # meses por período
MAX_VAR = 5.0                                             # variaciones interanuales > 500 % se descartan
CAP_SERIES_FAM = 10                                       # máximo de series por familia
PERIODOS_HASTA = pd.Timestamp("2026-06-01")               # último período que se puede pedir como "sin datos"
CUPOS = {"train": (300, 40), "val": (80, 10), "test": (150, 20)}   # (series con datos, series sin datos)
SEMILLA_SELECCION = 7


def datos(resp: dict):
    return (resp["json"] or {}).get("data")


def desplazar(fecha, anios: int) -> str:
    return (pd.to_datetime(fecha) + pd.DateOffset(years=anios)).strftime("%Y-%m-%d")


def sumar(fecha, frecuencia: str, k: int) -> pd.Timestamp:
    """Fecha `k` períodos después (k puede ser negativo)."""
    return pd.to_datetime(fecha) + pd.DateOffset(months=PASO[frecuencia] * k)


def periodo_al_azar(rng: random.Random, frecuencia: str) -> str:
    anio = rng.randint(2015, 2024)
    if frecuencia == "anual":
        return str(anio)
    mes = rng.choice([1, 4, 7, 10]) if frecuencia == "trimestral" else rng.randint(1, 12)
    return f"{anio}-{mes:02d}"


def fmt(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def nombre(fila) -> str:
    return str(fila["serie_titulo"]).replace("_", " ")


def unidad(fila) -> str:
    u = fila["serie_unidades"]
    return f" ({u})" if isinstance(u, str) and u.strip() else ""


def fila_spec(tipo: str, fila, params: dict, resp: dict, final: str, caso: str = "") -> dict:
    return {"tipo": tipo, "caso": caso, "serie_id": fila["serie_id"], "familia": fila["familia"],
            "frecuencia": fila["frecuencia"], "serie_titulo": fila["serie_titulo"],
            "serie_unidades": fila["serie_unidades"],
            "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "status": resp["status"], "data": json.dumps(datos(resp), ensure_ascii=False),
            "consultado": resp["consultado"], "respuesta_final": final}


def elegir(d: pd.DataFrame, n: int) -> pd.DataFrame:
    """Hasta `n` series, como mucho CAP_SERIES_FAM por familia, en orden aleatorio fijo."""
    return d.sample(frac=1, random_state=SEMILLA_SELECCION).groupby("familia").head(CAP_SERIES_FAM).head(n)


def seleccionar_series(catalogo: pd.DataFrame, split: str, n_con_datos: int, n_sin_datos: int):
    """(series con datos, series sin datos) de un split, solo de frecuencias soportadas."""
    base = catalogo[(catalogo["split"] == split) & catalogo["frecuencia"].isin(LAG)]
    con = base[base["estado"].isin(["normal", "api_adelantada"])]
    return elegir(con, n_con_datos), elegir(base[base["estado"] == "sin_datos"], n_sin_datos)


def _rng(serie_id: str, desplazamiento: int = 0) -> random.Random:
    """Generador determinista por serie: la misma serie produce siempre las mismas especificaciones."""
    return random.Random(zlib.crc32(serie_id.encode()) + desplazamiento)


def _variacion_interanual(fila, get, rng, n: int, desc: Counter) -> dict | None:
    """Variación interanual de un período al azar, verificada contra el cálculo propio con los valores crudos.

    Se descarta si algún valor crudo no es positivo (la variación no tiene sentido), si la API no coincide con
    v_t / v_(t-1) - 1 o si el resultado supera MAX_VAR.
    """
    sid, freq = fila["serie_id"], fila["frecuencia"]
    lag = LAG[freq]
    if n <= lag:
        return None
    pt = datos(get({"ids": sid, "limit": 1, "sort": "asc", "start": rng.randrange(lag, n)}))
    if not pt:
        return None
    d0 = pt[0][0]
    prev = desplazar(d0, -1)
    crudo = dict(datos(get({"ids": sid, "start_date": prev, "end_date": d0, "limit": 1000})) or [])
    v_t, v_prev = crudo.get(d0), crudo.get(prev)
    params = {"ids": sid, "start_date": prev, "end_date": d0, "representation_mode": "percent_change_a_year_ago"}
    rr = get(params)
    d = datos(rr)
    if not (rr["status"] == 200 and d and d[-1][0] == d0 and d[-1][1] is not None):
        desc["variacion: sin dato interanual"] += 1
    elif v_t is None or v_prev is None or v_t <= 0 or v_prev <= 0:
        desc["variacion: valor crudo no positivo o faltante"] += 1
    elif abs(d[-1][1] - (v_t / v_prev - 1)) > 1e-6:
        desc["variacion: la API no coincide con el cálculo propio"] += 1
    elif abs(d[-1][1]) > MAX_VAR:
        desc["variacion: valor absurdo (> 500%)"] += 1
    else:
        return fila_spec("variacion_interanual", fila, params, rr,
                         f"La variación interanual de {nombre(fila)} en {periodo(d0, freq)} fue de "
                         f"{d[-1][1] * 100:.2f}% (respecto de {periodo(prev, freq)}).")
    return None


def especificaciones_de_serie(fila, get: Callable[[dict], dict]) -> tuple[list[dict], Counter]:
    """Hasta cuatro especificaciones de una serie con datos y el conteo de motivos de descarte.

    El período de sin_datos cae antes de la primera observación o después de la última, sin pasar de
    PERIODOS_HASTA (no se pide un período que todavía no ocurrió) y verificando que la API devuelva vacío.
    """
    out, desc = [], Counter()
    sid, freq = fila["serie_id"], fila["frecuencia"]
    rng = _rng(sid)
    nom, uni = nombre(fila), unidad(fila)

    r0 = get({"ids": sid, "limit": 1, "sort": "asc"})
    rl = get({"ids": sid, "limit": 1, "sort": "desc"})
    if not (r0["status"] == 200 and datos(r0)):
        desc["la serie no devolvió datos al verificar"] += 1
        return out, desc
    n = r0["json"]["count"]
    primera = datos(r0)[0][0]
    ultimo = datos(rl)[0] if (rl["status"] == 200 and datos(rl)) else None
    if ultimo is None:                                   # respaldo si sort=desc devuelve vacío
        alt = datos(get({"ids": sid, "limit": 1, "sort": "asc", "start": n - 1}))
        ultimo = alt[0] if alt else None
    if ultimo is None or ultimo[1] is None:
        desc["no se pudo leer el último dato"] += 1
        return out, desc

    out.append(fila_spec("ultimo_dato", fila, {"ids": sid, "limit": 1, "sort": "desc"}, rl,
                         f"El último dato disponible de {nom}{uni} es {fmt(ultimo[1])}, de la fecha {ultimo[0]}."))

    pt = datos(get({"ids": sid, "limit": 1, "sort": "asc", "start": rng.randrange(n)}))
    if pt:
        p = periodo(pt[0][0], freq)
        params = {"ids": sid, "start_date": p, "end_date": p}
        rr = get(params)
        d = datos(rr)
        if rr["status"] == 200 and d and d[0][1] is not None:
            out.append(fila_spec("valor_puntual", fila, params, rr, f"{nom}{uni}: {fmt(d[0][1])} en {p}."))
        else:
            desc["valor_puntual: sin valor"] += 1

    var = _variacion_interanual(fila, get, rng, n, desc)
    if var:
        out.append(var)

    caso = "fuera_de_rango_despues" if rng.random() < 0.5 else "fuera_de_rango_antes"
    p = None
    if caso == "fuera_de_rango_despues":
        cand = [d for d in (sumar(ultimo[0], freq, k) for k in range(1, 7))
                if d <= PERIODOS_HASTA and (freq != "anual" or d.year <= 2025)]
        if cand:
            p = periodo(rng.choice(cand).strftime("%Y-%m-%d"), freq)
        else:
            caso = "fuera_de_rango_antes"            # la serie termina muy cerca de hoy
    if p is None:
        p = periodo(desplazar(primera, -rng.choice([1, 2, 3, 5])), freq)
    params = {"ids": sid, "start_date": p, "end_date": p}
    rr = get(params)
    if rr["status"] == 200 and datos(rr) == []:
        out.append(fila_spec("sin_datos", fila, params, rr,
                             f"La API no devuelve datos de {nom} para el período {p}. "
                             f"La serie tiene datos entre {periodo(primera, freq)} y {periodo(ultimo[0], freq)}.",
                             caso=caso))
    else:
        desc["sin_datos: la API devolvió datos o error"] += 1
    return out, desc


def especificacion_de_serie_vacia(fila, get: Callable[[dict], dict]) -> tuple[list[dict], Counter]:
    """Serie que la API devuelve vacía: un período al azar, verificado vacío."""
    out, desc = [], Counter()
    p = periodo_al_azar(_rng(fila["serie_id"]), fila["frecuencia"])
    params = {"ids": fila["serie_id"], "start_date": p, "end_date": p}
    rr = get(params)
    if rr["status"] == 200 and datos(rr) == []:
        out.append(fila_spec("sin_datos", fila, params, rr,
                             f"La API no devuelve datos para la serie {nombre(fila)}.", caso="serie_sin_datos"))
    else:
        desc["serie_sin_datos: la API devolvió datos o error"] += 1
    return out, desc

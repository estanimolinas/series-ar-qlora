"""Evaluador de respuestas del agente.

Una respuesta es correcta si termina con submit_answer, la última serie consultada es la objetivo,
el valor coincide (con tolerancia) y la fecha coincide en el nivel de la frecuencia de la serie.
Validado con un agente oráculo (100 %) y con controles negativos (~0 %).
"""
from __future__ import annotations

import numpy as np

TIPOS = ("ultimo_dato", "valor_puntual", "variacion_interanual", "sin_datos")


def periodo(fecha, frecuencia) -> str:
    """'2020-03-01' -> '2020-03' (o '2020' en series anuales)."""
    f = "" if fecha is None else str(fecha)
    return f[:4] if frecuencia == "anual" else f[:7]


def valor_fecha(tipo, frecuencia, datos, params):
    """Valor y fecha esperados a partir de la respuesta cruda de la API (`datos` = lista [fecha, valor])."""
    if tipo == "sin_datos" or not datos or datos[0][1] is None or datos[-1][1] is None:
        return None, None
    if tipo == "ultimo_dato":
        return datos[0][1], datos[0][0]
    if tipo == "valor_puntual":
        return datos[0][1], params["start_date"]
    return round(datos[-1][1] * 100, 2), datos[-1][0]      # variación: fracción -> porcentaje


def cerca(a, b, tol_abs) -> bool:
    return abs(a - b) <= max(tol_abs, 1e-3 * abs(b))


def puntuar(tipo, frecuencia, gold, submit, ids_ultima, objetivo) -> dict:
    """Puntúa un episodio. `gold` = {"valor", "fecha"}; `submit` = argumentos de submit_answer o None."""
    r = {"termino": submit is not None, "serie_ok": ids_ultima == objetivo}
    if submit is None:
        r.update(valor_ok=False, valor_flex=False, fecha_ok=False)
    elif tipo == "sin_datos":
        nulo = submit.get("valor") is None
        r.update(valor_ok=nulo, valor_flex=nulo, fecha_ok=submit.get("fecha") is None)
    else:
        v = submit.get("valor")
        es_num = isinstance(v, (int, float)) and not isinstance(v, bool)
        tol = 0.011 if tipo == "variacion_interanual" else 5e-5
        ok_estricto = bool(es_num and cerca(v, gold["valor"], tol))
        # "flex": en variaciones se acepta también la fracción (0.05 en vez de 5)
        ok_flex = ok_estricto or bool(
            tipo == "variacion_interanual" and es_num and cerca(v, gold["valor"] / 100, 1e-4))
        r.update(valor_ok=ok_estricto, valor_flex=ok_flex,
                 fecha_ok=submit.get("fecha") is not None
                 and periodo(submit["fecha"], frecuencia) == periodo(gold["fecha"], frecuencia))
    r["correcta"] = bool(r["termino"] and r["serie_ok"] and r["valor_ok"] and r["fecha_ok"])
    r["correcta_flex"] = bool(r["termino"] and r["serie_ok"] and r["valor_flex"] and r["fecha_ok"])
    return r


def causa(r) -> str:
    """Primera causa de fallo, en orden de gravedad."""
    if r["correcta"]:
        return "correcta"
    if not r["termino"]:
        return "no_termino"
    if not r["serie_ok"]:
        return "serie_equivocada"
    if not r["valor_ok"]:
        return "convencion_porcentaje" if r["valor_flex"] else "valor_equivocado"
    return "fecha_equivocada"


def wilson(k, n, z=1.96):
    """Intervalo de Wilson para una proporción k/n."""
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    mitad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, float(centro - mitad)), min(1.0, float(centro + mitad))   # recorte: error de redondeo en k=0 o k=n


def ids_consultados(mensajes) -> list[str]:
    """Series que el agente pidió con get_series, en orden (a partir de los mensajes del episodio)."""
    ids = []
    for m in mensajes:
        if m.get("role") != "assistant":
            continue
        for t in m.get("tool_calls") or []:
            if t["function"]["name"] == "get_series":
                ids.append(str(t["function"]["arguments"].get("ids")))
    return ids


def puntuar_alguna_serie(r: dict, ids: list[str], objetivo: str) -> dict:
    """Variante para análisis de sensibilidad: la serie cuenta como correcta si la objetivo está entre TODAS las consultadas.

    La métrica principal (`puntuar`) mira solo la última consulta a get_series. Esta variante no penaliza a un agente
    que consultó varias series y respondió con una anterior. Supone que, si el valor y la fecha coinciden con el dato
    esperado, la respuesta salió de la serie objetivo: no puede comprobar de cuál de las consultadas salió.
    `r` es el resultado ya puntuado (con termino, valor_ok y fecha_ok).
    """
    alguna = objetivo in ids
    return {"serie_ok_alguna": alguna,
            "correcta_alguna": bool(r["termino"] and alguna and r["valor_ok"] and r["fecha_ok"])}

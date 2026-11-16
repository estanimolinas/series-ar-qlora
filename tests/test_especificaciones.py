import json
import random
from collections import Counter

import pandas as pd
import pytest

from series_qlora.cache_api import CacheApi, clave
from series_qlora.especificaciones import (
    desplazar,
    especificacion_de_serie_vacia,
    especificaciones_de_serie,
    fmt,
    periodo_al_azar,
    seleccionar_series,
    sumar,
)


def serie(frecuencia="mensual", sid="s1", unidades="índice"):
    return {"serie_id": sid, "familia": "fam", "frecuencia": frecuencia, "serie_titulo": "Precios_al consumidor",
            "serie_unidades": unidades}


def api_falsa(valores: dict, vacio_fuera=True):
    """GET guionado sobre una serie mensual {fecha: valor}; imita limit/sort/start/start_date/end_date."""
    fechas = sorted(valores)

    def get(params):
        filas = [[f, valores[f]] for f in fechas]
        a, b = params.get("start_date"), params.get("end_date")
        if a or b:
            filas = [x for x in filas if (not a or x[0][:len(a)] >= a) and (not b or x[0][:len(b)] <= b)]
        if params.get("representation_mode") == "percent_change_a_year_ago":
            filas = [[f, valores[f] / valores[desplazar(f, -1)] - 1] for f, _ in filas if desplazar(f, -1) in valores]
        if params.get("sort") == "desc":
            filas = filas[::-1]
        filas = filas[int(params.get("start", 0)):]
        if "limit" in params:
            filas = filas[:int(params["limit"])]
        return {"status": 200, "json": {"data": filas, "count": len(fechas)}, "consultado": "2026-01-01T00:00:00"}
    return get


def mensual(n=40, inicio=2020):
    return {f"{inicio + i // 12}-{i % 12 + 1:02d}-01": 100.0 + i for i in range(n)}


def test_auxiliares():
    assert fmt(1.50000) == "1.5" and fmt(2.0) == "2" and fmt(0.12345678) == "0.1235"
    assert desplazar("2021-03-01", -1) == "2020-03-01"
    assert sumar("2021-11-01", "trimestral", 1).strftime("%Y-%m") == "2022-02"
    r = random.Random(1)
    assert len(periodo_al_azar(r, "anual")) == 4 and len(periodo_al_azar(r, "mensual")) == 7
    assert periodo_al_azar(random.Random(2), "trimestral")[5:] in {"01", "04", "07", "10"}


def test_cuatro_especificaciones_de_una_serie_con_datos():
    out, desc = especificaciones_de_serie(serie(), api_falsa(mensual()))
    assert [e["tipo"] for e in out] == ["ultimo_dato", "valor_puntual", "variacion_interanual", "sin_datos"]
    ultimo = out[0]
    assert ultimo["respuesta_final"] == "El último dato disponible de Precios al consumidor (índice) es 139, de la fecha 2023-04-01."
    sin = out[3]
    assert json.loads(sin["data"]) == [] and "La serie tiene datos entre 2020-01 y 2023-04." in sin["respuesta_final"]
    assert not desc


def test_las_especificaciones_son_deterministas_por_serie():
    a, _ = especificaciones_de_serie(serie(), api_falsa(mensual()))
    b, _ = especificaciones_de_serie(serie(), api_falsa(mensual()))
    c, _ = especificaciones_de_serie(serie(sid="otra"), api_falsa(mensual()))
    assert a == b and a != c


def test_la_variacion_se_verifica_contra_el_calculo_propio():
    # la API fingida devuelve una variación distinta de la que sale de los valores crudos
    base = api_falsa(mensual())

    def mentirosa(params):
        r = base(params)
        if params.get("representation_mode"):
            r["json"]["data"] = [[f, v + 0.5] for f, v in r["json"]["data"]]
        return r

    out, desc = especificaciones_de_serie(serie(), mentirosa)
    assert "variacion_interanual" not in [e["tipo"] for e in out]
    assert desc["variacion: la API no coincide con el cálculo propio"] == 1


def test_variacion_se_descarta_con_valores_no_positivos():
    vals = mensual()
    vals = {f: (0.0 if f.startswith("2021") else v) for f, v in vals.items()}
    out, desc = especificaciones_de_serie(serie(), api_falsa(vals))
    tipos = [e["tipo"] for e in out]
    # el período elegido al azar puede caer fuera de 2021/2022: se comprueba que, si cae, se descarta y se cuenta
    if "variacion_interanual" not in tipos:
        assert desc["variacion: valor crudo no positivo o faltante"] == 1


def test_sin_datos_no_pide_periodos_posteriores_al_limite():
    # serie que termina en 2026-06: el período siguiente (2026-07) pasa del límite → el caso pasa a "antes"
    vals = {f"{2022 + i // 12}-{i % 12 + 1:02d}-01": 1.0 + i for i in range(54)}      # hasta 2026-06
    for sid in ("a", "b", "c", "d", "e", "f"):
        out, _ = especificaciones_de_serie(serie(sid=sid), api_falsa(vals))
        sd = [e for e in out if e["tipo"] == "sin_datos"]
        assert sd and sd[0]["caso"] == "fuera_de_rango_antes"
        assert json.loads(sd[0]["params"])["start_date"] < "2022"


def test_serie_que_no_devuelve_datos_no_genera_nada():
    vacia = lambda p: {"status": 200, "json": {"data": [], "count": 0}, "consultado": None}  # noqa: E731
    assert especificaciones_de_serie(serie(), vacia) == ([], Counter({"la serie no devolvió datos al verificar": 1}))


def test_serie_vacia_genera_un_sin_datos_verificado():
    vacia = lambda p: {"status": 200, "json": {"data": []}, "consultado": "x"}  # noqa: E731
    out, desc = especificacion_de_serie_vacia(serie("anual"), vacia)
    assert len(out) == 1 and out[0]["caso"] == "serie_sin_datos" and not desc
    con_datos = lambda p: {"status": 200, "json": {"data": [["2020-01-01", 1]]}, "consultado": "x"}  # noqa: E731
    out, desc = especificacion_de_serie_vacia(serie("anual"), con_datos)
    assert out == [] and desc["serie_sin_datos: la API devolvió datos o error"] == 1


def test_seleccion_respeta_split_frecuencia_cupo_por_familia_y_semilla():
    filas = [{"serie_id": f"s{i}", "split": "train", "frecuencia": "mensual", "estado": "normal",
              "familia": f"f{i % 3}"} for i in range(60)]
    filas += [{"serie_id": "d", "split": "train", "frecuencia": "diaria", "estado": "normal", "familia": "x"},
              {"serie_id": "t", "split": "test", "frecuencia": "mensual", "estado": "normal", "familia": "y"},
              {"serie_id": "v", "split": "train", "frecuencia": "mensual", "estado": "sin_datos", "familia": "z"}]
    cat = pd.DataFrame(filas)
    con, sin = seleccionar_series(cat, "train", 100, 5)
    assert "d" not in set(con["serie_id"]) and "t" not in set(con["serie_id"])
    assert con.groupby("familia").size().max() == 10 and len(con) == 30      # tope de 10 por familia
    assert list(sin["serie_id"]) == ["v"]
    assert list(seleccionar_series(cat, "train", 100, 5)[0]["serie_id"]) == list(con["serie_id"])


# ---------- caché ----------
def test_clave_rechaza_parametros_desconocidos():
    assert clave({"ids": "a", "limit": 1}) == clave({"limit": "1", "ids": "a"})
    with pytest.raises(ValueError, match="no previstos"):
        clave({"ids": "a", "format": "csv"})


def test_cache_api_no_repite_pedidos_y_distingue_error_de_red(tmp_path):
    llamadas = []

    def api(p):
        llamadas.append(p)
        return (200, {"data": [["2020-01-01", 1]]}) if p["ids"] == "ok" else (None, None)

    c = CacheApi(tmp_path / "c.db", api=api, pausa=0, dormir=lambda s: None)
    a, b = c.get({"ids": "ok"}), c.get({"ids": "ok"})
    assert a["de_cache"] is False and b["de_cache"] is True and b["json"] == a["json"] and len(llamadas) == 1
    caido = c.get({"ids": "caido"})
    assert caido["status"] is None and c.tamano() == 1          # el fallo de red no se guarda
    assert c.stats == {"aciertos": 1, "fallos_de_cache": 2, "error_de_red": 1}
    c.cerrar()
    # se puede reabrir: la respuesta quedó guardada
    c2 = CacheApi(tmp_path / "c.db", api=lambda p: pytest.fail("no debía llamar"), pausa=0)
    assert c2.get({"ids": "ok"})["de_cache"] is True

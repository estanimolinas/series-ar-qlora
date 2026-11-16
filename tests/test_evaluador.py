import pytest

from series_qlora.evaluador import causa, cerca, periodo, puntuar, valor_fecha, wilson

PARAMS = {"ids": "ser_a", "start_date": "2020-03"}


def test_periodo_respeta_la_frecuencia():
    assert periodo("2020-03-01", "mensual") == "2020-03"
    assert periodo("2020-03-01", "anual") == "2020"
    assert periodo(None, "mensual") == ""


def test_valor_fecha_por_tipo():
    datos = [["2021-05-01", 10.0], ["2021-04-01", 9.0]]
    assert valor_fecha("ultimo_dato", "mensual", datos, PARAMS) == (10.0, "2021-05-01")
    assert valor_fecha("valor_puntual", "mensual", datos, PARAMS) == (10.0, "2020-03")
    assert valor_fecha("variacion_interanual", "mensual", [["a", 0.0], ["2021-05-01", 0.1234]], PARAMS) == (12.34, "2021-05-01")
    assert valor_fecha("sin_datos", "mensual", datos, PARAMS) == (None, None)
    assert valor_fecha("ultimo_dato", "mensual", [], PARAMS) == (None, None)
    assert valor_fecha("ultimo_dato", "mensual", [["2021-05-01", None]], PARAMS) == (None, None)


def test_cerca_usa_la_tolerancia_mayor():
    assert cerca(100.05, 100.0, 5e-5)          # 1e-3 relativo = 0.1
    assert not cerca(100.2, 100.0, 5e-5)
    assert cerca(0.00004, 0.0, 5e-5)


def correcta(**cambios):
    base = dict(tipo="valor_puntual", freq="mensual", gold={"valor": 10.0, "fecha": "2020-03"},
                sub={"valor": 10.0, "fecha": "2020-03"}, ids="ser_a", objetivo="ser_a")
    base.update(cambios)
    return puntuar(base["tipo"], base["freq"], base["gold"], base["sub"], base["ids"], base["objetivo"])


def test_respuesta_correcta():
    r = correcta()
    assert r["correcta"] and causa(r) == "correcta"


@pytest.mark.parametrize("cambios,esperada", [
    (dict(sub=None), "no_termino"),
    (dict(ids="ser_b"), "serie_equivocada"),
    (dict(sub={"valor": 11.0, "fecha": "2020-03"}), "valor_equivocado"),
    (dict(sub={"valor": 10.0, "fecha": "2020-04"}), "fecha_equivocada"),
    (dict(sub={"valor": "10", "fecha": "2020-03"}), "valor_equivocado"),   # un string no es un número
    (dict(sub={"valor": True, "fecha": "2020-03"}), "valor_equivocado"),   # un bool tampoco
    (dict(sub={"valor": 10.0, "fecha": None}), "fecha_equivocada"),
])
def test_causas_de_fallo(cambios, esperada):
    r = correcta(**cambios)
    assert not r["correcta"] and causa(r) == esperada


def test_fecha_anual_compara_solo_el_anio():
    r = correcta(freq="anual", gold={"valor": 10.0, "fecha": "2020-01-01"}, sub={"valor": 10.0, "fecha": "2020"})
    assert r["correcta"]


def test_variacion_en_fraccion_solo_es_correcta_en_modo_flex():
    gold = {"valor": 5.0, "fecha": "2021-05-01"}
    r = correcta(tipo="variacion_interanual", gold=gold, sub={"valor": 0.05, "fecha": "2021-05"})
    assert not r["correcta"] and r["correcta_flex"] and causa(r) == "convencion_porcentaje"


def test_variacion_en_porcentaje_tolera_centesimas():
    gold = {"valor": 5.0, "fecha": "2021-05-01"}
    assert correcta(tipo="variacion_interanual", gold=gold, sub={"valor": 5.01, "fecha": "2021-05"})["correcta"]
    assert not correcta(tipo="variacion_interanual", gold=gold, sub={"valor": 5.2, "fecha": "2021-05"})["correcta"]


def test_sin_datos_exige_valor_y_fecha_nulos():
    gold = {"valor": None, "fecha": None}
    assert correcta(tipo="sin_datos", gold=gold, sub={"valor": None, "fecha": None})["correcta"]
    assert not correcta(tipo="sin_datos", gold=gold, sub={"valor": 3.0, "fecha": None})["correcta"]
    assert not correcta(tipo="sin_datos", gold=gold, sub={"valor": None, "fecha": "2020-01"})["correcta"]


def test_wilson_contiene_la_proporcion_y_se_achica_con_n():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi
    lo2, hi2 = wilson(500, 1000)
    assert (hi2 - lo2) < (hi - lo)
    assert 0 <= wilson(0, 10)[0] and wilson(10, 10)[1] <= 1

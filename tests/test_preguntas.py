import json

import pandas as pd

from series_qlora.preguntas import (
    armar_final,
    armar_prompt,
    costo_usd,
    defecto_periodo,
    generar_pregunta,
    leer_fidelidad,
    marca_var,
    periodo_de_spec,
    periodo_humano,
    validar,
)


def spec(tipo="valor_puntual", freq="mensual", params=None, titulo="Exportaciones de bienes por rubro y destino"):
    params = params or {"ids": "s", "start_date": "2021-03", "end_date": "2021-03"}
    return {"tipo": tipo, "frecuencia": freq, "params": json.dumps(params), "serie_titulo": titulo,
            "serie_id": "s", "spec_id": 1}


SERIE = pd.Series({"serie_titulo": "Exportaciones", "serie_descripcion": None, "serie_unidades": "USD",
                   "frecuencia": "mensual", "dataset_titulo": "Balanza", "distribucion_titulo": "Comercio"})


def test_periodo_de_spec_y_periodo_humano():
    assert periodo_de_spec(spec()) == "2021-03"
    assert periodo_de_spec(spec("ultimo_dato")) is None
    var = spec("variacion_interanual", params={"ids": "s", "start_date": "2020-03", "end_date": "2021-03"})
    assert periodo_de_spec(var) == "2021-03"
    anual = spec("variacion_interanual", "anual", {"ids": "s", "start_date": "2019", "end_date": "2020"})
    assert periodo_de_spec(anual) == "2020"
    assert periodo_humano("mensual", "2021-03") == "marzo de 2021"
    assert periodo_humano("trimestral", "2021-07") == "tercer trimestre de 2021"
    assert periodo_humano("anual", "2021") == "2021"


def test_validar_acepta_una_pregunta_correcta():
    assert validar(spec(), "¿Cuánto valieron las ventas externas en marzo de 2021?", "coloquial", "2021-03") == ""


def test_validar_rechaza_cada_defecto():
    r, p = spec(), "2021-03"
    assert validar(r, "", "coloquial", p) == "vacía"
    assert validar(r, "ventas afuera", "coloquial", p).startswith("largo")
    assert validar(r, "¿Cuánto se vendió afuera en marzo de 2020?", "coloquial", p) == "falta el año"
    assert validar(r, "¿Cuánto se vendió afuera en marzo de 2021 y 2020?", "coloquial", p) == "año de más"
    assert validar(r, "¿Cuánto se vendió afuera en marzo de 2021 al 5 %?", "coloquial", p) == "tiene otros números"
    assert validar(r, "¿Cuánto se vendió afuera durante 2021 en total?", "coloquial", p) == "falta el mes"
    assert validar(spec(freq="trimestral"), "¿Cuánto se vendió afuera en 2021 en total?", "coloquial",
                   "2021-04") == "falta 'trimestre'"
    assert validar(r, "exportaciones de bienes por rubro y destino en marzo de 2021 cuánto", "tecnica",
                   p) == "copia el título"
    assert validar(spec("ultimo_dato"), "¿Cuál es el último dato de 2021?", "coloquial", None) == \
        "tiene números y no debería"


def test_validar_variacion_exige_interanual_y_acepta_el_anio_previo():
    r = spec("variacion_interanual", params={"ids": "s", "start_date": "2020-03", "end_date": "2021-03"})
    assert validar(r, "¿Cuánto subió respecto de marzo de 2020 en marzo de 2021?", "coloquial",
                   "2021-03") == "no aclara que es interanual"
    assert validar(r, "¿Cuál fue la variación interanual en marzo de 2021 frente a 2020?", "coloquial",
                   "2021-03") == ""


def test_setiembre_vale_como_septiembre():
    r = spec(params={"ids": "s", "start_date": "2021-09", "end_date": "2021-09"})
    assert validar(r, "¿Cuánto se vendió afuera en setiembre de 2021?", "coloquial", "2021-09") == ""


def test_prompt_incluye_el_periodo_y_no_filtra_otros_datos():
    pr = armar_prompt(spec(), "2021-03", SERIE)
    assert "marzo de 2021" in pr and "Título técnico: Exportaciones" in pr and "Descripción: \n" in pr
    assert "ninguna fecha" in armar_prompt(spec("ultimo_dato"), None, SERIE)
    assert "variación interanual" in armar_prompt(spec("variacion_interanual"), "2021-03", SERIE)


class Llamador:
    def __init__(self, respuestas):
        self.respuestas, self.llamadas = list(respuestas), []

    def __call__(self, prompt, max_tokens):
        self.llamadas.append(max_tokens)
        r = self.respuestas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


BUENA = json.dumps({"coloquial": "¿Cuánto se vendió afuera en marzo de 2021?",
                    "tecnica": "¿Cuál fue el valor de las exportaciones de bienes en dólares durante marzo de 2021?"})


def test_generar_pregunta_acepta_al_primer_intento():
    out = generar_pregunta(spec(), SERIE, Llamador([(BUENA, False, 100, 40)]))
    assert out["ok"] and out["intentos"] == 1 and (out["ent"], out["sal"]) == (100, 40) and out["motivos"] == ""


def test_generar_pregunta_reintenta_y_acumula_motivos_y_tokens():
    mala = json.dumps({"coloquial": "ventas afuera", "tecnica": "x"})
    ll = Llamador([("no es json", False, 10, 5), (mala, False, 10, 5), ("{", True, 10, 5), (BUENA, False, 10, 5)])
    out = generar_pregunta(spec(), SERIE, ll, intentos=4)
    assert out["ok"] and out["intentos"] == 4 and out["ent"] == 40 and out["sal"] == 20
    assert out["motivos"].startswith("JSON inválido | coloquial: largo (2 palabras) | tecnica: largo (1 palabras)")
    assert "max_tokens" in out["motivos"]
    assert ll.llamadas == [3000, 3000, 3000, 6000]           # el último intento sube el tope


def test_generar_pregunta_se_rinde_y_registra_el_error():
    ll = Llamador([("no es json", False, 1, 1)] * 3)
    out = generar_pregunta(spec(), SERIE, ll)
    assert not out["ok"] and out["error"] == "sin respuesta válida tras 3 intentos" and out["intentos"] == 3
    out = generar_pregunta(spec(), SERIE, Llamador([RuntimeError("caída")]))
    assert not out["ok"] and out["error"] == "RuntimeError: caída"


def test_defectos_de_periodo():
    assert defecto_periodo("anual", "valor_puntual", "¿Cuánto fue en marzo de 2020?") == "mes o trimestre en serie anual"
    assert defecto_periodo("anual", "valor_puntual", "¿Cuánto fue en el primer trimestre de 2020?") != ""
    assert defecto_periodo("trimestral", "valor_puntual", "Valor en setiembre de 2020") == "mes en serie trimestral"
    assert defecto_periodo("mensual", "valor_puntual", "Valor en el tercer trimestre") == "trimestre en serie mensual"
    assert defecto_periodo("mensual", "ultimo_dato", "¿Cuál es el dato de enero?") == "fecha en pregunta de último dato"
    assert defecto_periodo("mensual", "valor_puntual", "Valor en marzo de 2021") == ""
    assert defecto_periodo("mensual", "valor_puntual", "Valor de la mareada de 2021") == ""     # no es un mes


def test_marca_var():
    assert marca_var("valor_puntual", "var_ia", "%") == ""
    assert marca_var("variacion_interanual", "ipc_var_mensual", "índice") == "var_sobre_tasa"
    assert marca_var("variacion_interanual", "Salario", "variación porcentual") == "var_sobre_tasa"
    assert marca_var("variacion_interanual", "Desempleo", "%") == "var_sobre_porcentaje"
    assert marca_var("variacion_interanual", "Tasa de empleo", "índice") == "var_sobre_porcentaje"
    assert marca_var("variacion_interanual", "Exportaciones", "USD") == ""


def test_leer_fidelidad():
    t = 'Resultado: {"coloquial": {"ok": false, "motivo": "otra medida"}, "tecnica": {"ok": true}} fin'
    assert leer_fidelidad(t) == {"col_ok": False, "col_motivo": "otra medida", "tec_ok": True, "tec_motivo": ""}


def test_costo_usd():
    assert costo_usd(1_000_000, 1_000_000) == 0.60


def test_armar_final_excluye_marca_y_no_pierde_filas():
    cat = pd.DataFrame({"serie_id": ["a", "b", "c", "d"], "serie_titulo": ["Exportaciones", "Tasa de paro", "X", "Y"],
                        "serie_unidades": ["USD", "%", "USD", "USD"]}).set_index("serie_id")
    base = {"frecuencia": "mensual", "tecnica": "tecnica correcta"}
    q = pd.DataFrame([
        {**base, "spec_id": 0, "serie_id": "a", "tipo": "valor_puntual", "coloquial": "ventas afuera"},
        {**base, "spec_id": 1, "serie_id": "a", "tipo": "valor_puntual", "coloquial": "Ventas afuera "},   # repetida
        {**base, "spec_id": 2, "serie_id": "b", "tipo": "variacion_interanual", "coloquial": "paro interanual"},
        {**base, "spec_id": 3, "serie_id": "c", "tipo": "valor_puntual", "coloquial": "otra cosa"},     # técnica marcada
        {**base, "spec_id": 4, "serie_id": "d", "tipo": "valor_puntual", "coloquial": "dato del tercer trimestre"},
    ])
    fid = pd.DataFrame({"spec_id": [0, 1, 2, 3, 4], "col_ok": [True, True, False, True, True],
                        "col_motivo": "", "tec_ok": [True, True, True, False, True], "tec_motivo": ""})
    fin, exc = armar_final(q, fid, cat)
    assert len(fin) + len(exc) == len(q)
    assert set(exc["spec_id"]) == {3, 4}                      # técnica marcada; trimestre en serie mensual
    f = fin.set_index("spec_id")
    assert f.loc[0, "coloquial_ambigua"] and f.loc[1, "coloquial_ambigua"]      # repetidas
    assert f.loc[2, "coloquial_ambigua"] and f.loc[2, "marca_var"] == "var_sobre_porcentaje"
    assert "cat_serie_titulo" not in fin and "def_periodo" not in fin


def test_armar_final_exige_verificacion_completa():
    import pytest
    cat = pd.DataFrame({"serie_id": ["a"], "serie_titulo": ["x"], "serie_unidades": ["u"]}).set_index("serie_id")
    q = pd.DataFrame([{"spec_id": 0, "serie_id": "a", "tipo": "valor_puntual", "frecuencia": "mensual",
                       "coloquial": "c", "tecnica": "t"}])
    vacia = pd.DataFrame({"spec_id": [], "col_ok": [], "col_motivo": [], "tec_ok": [], "tec_motivo": []})
    with pytest.raises(AssertionError, match="Faltan verificaciones"):
        armar_final(q, vacia, cat)
    fin, exc = armar_final(q, vacia, cat, exigir_verificacion=False)
    assert len(fin) == 0 and len(exc) == 1                    # sin verificar = excluida

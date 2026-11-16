import json
from types import SimpleNamespace

import pandas as pd
import pytest

from series_qlora import config
from series_qlora.esquema import HERRAMIENTAS
from series_qlora.trayectorias import (
    argumentos_submit,
    armar_trayectoria,
    construir,
    costo_usd,
    extraer_consulta,
    pedir_consulta,
    preguntas_de_entrenamiento,
)

DATOS_VAR = [["2019-03-01", 0.5], ["2020-03-01", 0.0512]]


def fila(tipo="valor_puntual", **extra):
    base = {"qid": "7-tecnica", "spec_id": 7, "tipo": tipo, "estilo": "tecnica", "serie_id": "ser_emae",
            "familia": "F1", "frecuencia": "mensual", "pregunta": "¿Cuánto fue?",
            "respuesta_final": "Fue de 3. La serie tiene datos entre 2000 y 2020.",
            "serie_titulo": "emae", "serie_unidades": None,
            "data": json.dumps([["2020-03-01", 3.0]]), "params": json.dumps({"ids": "ser_emae", "start_date": "2020-03",
                                                                           "end_date": "2020-03"})}
    return {**base, **extra}


CANDIDATOS = json.dumps([{"serie_id": "otra", "titulo": "x", "distribucion": "d", "frecuencia": "mensual"},
                         {"serie_id": "ser_emae", "titulo": "emae", "distribucion": "d", "frecuencia": "mensual"}],
                        ensure_ascii=False)


def test_esquema_de_get_series_coincide_con_los_parametros_permitidos():
    por_nombre = {h["function"]["name"]: h["function"] for h in HERRAMIENTAS}
    assert set(por_nombre) == {"search_series", "get_series", "submit_answer"}
    assert set(por_nombre["get_series"]["parameters"]["properties"]) == config.PARAMS_TOOL
    json.dumps(HERRAMIENTAS)


@pytest.mark.parametrize("tipo,esperado", [
    ("ultimo_dato", {"valor": 3.0, "fecha": "2020-03-01"}),
    ("valor_puntual", {"valor": 3.0, "fecha": "2020-03"}),
    ("sin_datos", {"valor": None, "fecha": None}),
])
def test_argumentos_submit_por_tipo(tipo, esperado):
    f = fila(tipo)
    a = argumentos_submit(f, json.loads(f["data"]), json.loads(f["params"]))
    assert {k: a[k] for k in ("valor", "fecha")} == esperado


def test_sin_datos_quita_la_aclaracion_de_rango_de_la_respuesta():
    f = fila("sin_datos")
    assert argumentos_submit(f, [], {})["respuesta"] == "Fue de 3."


def test_variacion_pasa_de_fraccion_a_porcentaje_y_la_fecha_al_nivel_de_la_frecuencia():
    a = argumentos_submit(fila("variacion_interanual"), DATOS_VAR, {})
    assert a["valor"] == 5.12 and a["fecha"] == "2020-03"
    assert argumentos_submit(fila("variacion_interanual", frecuencia="anual"), DATOS_VAR, {})["fecha"] == "2020"


def test_trayectoria_tiene_la_forma_que_espera_el_entrenamiento():
    t = armar_trayectoria(fila(), "emae actividad", CANDIDATOS)
    m = t["messages"]
    assert [x["role"] for x in m] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    llamadas = [x["tool_calls"][0]["function"]["name"] for x in m if x.get("tool_calls")]
    assert llamadas == ["search_series", "get_series", "submit_answer"]
    assert m[1]["tool_calls"][0]["function"]["arguments"] == {"query": "emae actividad"}
    assert m[2]["content"] == CANDIDATOS                       # lo que devolvió el buscador, sin retocar
    assert json.loads(m[4]["content"])["data"] == [["2020-03-01", 3.0]]
    assert t["rango_busqueda"] == 2 and t["familia"] == "F1"


def test_si_la_serie_no_esta_entre_los_candidatos_no_hay_trayectoria():
    assert armar_trayectoria(fila(serie_id="inexistente"), "q", CANDIDATOS) is None


def test_construir_cuenta_las_preguntas_perdidas_por_motivo():
    Q = pd.DataFrame([fila(qid="1-tecnica"), fila(qid="2-coloquial", estilo="coloquial"),
                      fila(qid="3-tecnica", serie_id="inexistente"), fila(qid="4-tecnica")])
    consultas = {"1-tecnica": "a", "2-coloquial": "b", "3-tecnica": "c"}
    t, perdidas = construir(Q, consultas, lambda q: CANDIDATOS)
    assert [x["qid"] for x in t] == ["1-tecnica", "2-coloquial"]
    assert perdidas == {"la serie no aparece en el top 10 (tecnica)": 1, "sin consulta generada": 1}


@pytest.mark.parametrize("texto,esperado", [
    ('{"consulta": "cheques cancelatorios pesos"}', "cheques cancelatorios pesos"),
    ('Acá va: {"consulta": " inflación mensual "} listo', "inflación mensual"),
    ('{"consulta": "inflación 2020"}', None),                                  # no admite dígitos
    ('{"consulta": "uno dos tres cuatro cinco seis siete ocho nueve"}', None),  # más de 8 palabras
    ('{"consulta": ""}', None),
    ("sin json", None),
])
def test_extraer_consulta(texto, esperado):
    assert extraer_consulta(texto) == esperado


class ClienteFalso:
    def __init__(self, respuestas):
        self.respuestas, self.llamadas = list(respuestas), 0
        self.messages = SimpleNamespace(create=self._crear)

    def _crear(self, **kw):
        self.llamadas += 1
        r = self.respuestas.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=r)],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=10))


@pytest.fixture
def catalogo_serie():
    return pd.DataFrame({"serie_titulo": ["emae"], "serie_descripcion": [None], "dataset_titulo": ["EMAE"]},
                        index=pd.Index(["ser_emae"], name="serie_id"))


def test_pedir_consulta_reintenta_si_la_respuesta_no_es_valida(catalogo_serie):
    c = ClienteFalso(['{"consulta": "emae 2020"}', '{"consulta": "actividad económica"}'])
    r = pedir_consulta(c, fila(), catalogo_serie)
    assert r["ok"] and r["consulta"] == "actividad económica" and c.llamadas == 2 and r["ent"] == 100


def test_pedir_consulta_registra_el_fallo_tras_agotar_los_intentos(catalogo_serie):
    r = pedir_consulta(ClienteFalso(["x"] * 3), fila(), catalogo_serie)
    assert not r["ok"] and r["error"] == "sin consulta válida"


def test_pedir_consulta_registra_errores_comunes_y_relanza_los_fatales(catalogo_serie):
    r = pedir_consulta(ClienteFalso([TimeoutError("lento")]), fila(), catalogo_serie)
    assert not r["ok"] and "TimeoutError" in r["error"]
    with pytest.raises(PermissionError):
        pedir_consulta(ClienteFalso([PermissionError("clave")]), fila(), catalogo_serie, errores_fatales=(PermissionError,))


def test_costo_usd():
    R = pd.DataFrame({"ent": [1_000_000, None], "sal": [2_000_000, None]})
    assert costo_usd(R) == pytest.approx(0.10 + 1.00)


def specs_y_finales(n=200):
    tipos = ["ultimo_dato", "valor_puntual", "variacion_interanual", "sin_datos"]
    E = pd.DataFrame({"data": ["[]"] * n, "respuesta_final": ["r"] * n, "serie_titulo": ["t"] * n,
                      "serie_unidades": ["u"] * n})
    F = pd.DataFrame({"spec_id": range(n), "tipo": [tipos[i % 4] if i % 3 else "sin_datos" for i in range(n)],
                      "marca_var": ["var_sobre_tasa" if i % 17 == 0 else None for i in range(n)],
                      "tecnica": [f"t{i}" for i in range(n)], "coloquial": [f"c{i}" for i in range(n)],
                      "coloquial_ambigua": [i % 5 == 0 for i in range(n)], "familia": ["F"] * n})
    return E, F


def test_preguntas_de_entrenamiento_aplica_los_filtros_y_la_proporcion_de_sin_datos():
    E, F = specs_y_finales()
    Q = preguntas_de_entrenamiento(E, F, pct_sin_datos=0.18)
    assert Q["qid"].is_unique
    assert not Q["marca_var"].eq("var_sobre_tasa").any()
    assert (Q[Q["tipo"] == "sin_datos"]["estilo"] == "tecnica").all()
    ambiguas = set(F[F["coloquial_ambigua"]]["spec_id"])
    assert not Q[(Q["estilo"] == "coloquial") & Q["spec_id"].isin(ambiguas)].shape[0]
    assert abs((Q["tipo"] == "sin_datos").mean() - 0.18) < 0.03
    pd.testing.assert_frame_equal(Q, preguntas_de_entrenamiento(E, F, pct_sin_datos=0.18))   # determinista

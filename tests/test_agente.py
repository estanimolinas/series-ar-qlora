import json

from conftest import llamada
from series_qlora.agente import episodio, parsear_llamada
from series_qlora.evaluador import puntuar


def guion(*textos):
    """Generador que devuelve los textos en orden."""
    it = iter(textos)
    return lambda msgs: (next(it), 100, 20, 0.1)


def test_parsear_llamada():
    assert parsear_llamada(llamada("search_series", query="ipc")) == ("search_series", {"query": "ipc"})
    assert parsear_llamada("no llamo a nada") is None


def test_parsear_llamada_invalida():
    for malo in ['<tool_call>{"name": "x"}</tool_call>',
                 '<tool_call>{"name": "x", "arguments": "texto"}</tool_call>',
                 "<tool_call>{no es json}</tool_call>"]:
        try:
            parsear_llamada(malo)
        except ValueError:
            continue
        raise AssertionError(f"debió fallar: {malo}")


def test_episodio_oraculo_resuelve_y_el_evaluador_lo_da_por_correcto(herramientas):
    generar = guion(
        llamada("search_series", query="actividad economica"),
        llamada("get_series", ids="ser_emae", limit=1, sort="desc"),
        llamada("submit_answer", respuesta="x", valor=1.5, fecha="2020-01"),
    )
    log, msgs = episodio("¿Último dato de la actividad?", None, generar, herramientas)
    assert (log["n_search"], log["n_get"], log["pasos"]) == (1, 1, 3)
    assert log["submit"]["valor"] == 1.5 and log["ids_ultima"] == "ser_emae"
    r = puntuar("ultimo_dato", "mensual", {"valor": 1.5, "fecha": "2020-01-01"}, log["submit"], log["ids_ultima"], "ser_emae")
    assert r["correcta"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]


def test_el_resultado_de_la_herramienta_vuelve_al_modelo(herramientas):
    vistos = []

    def generar(msgs):
        vistos.append(list(msgs))
        textos = [llamada("search_series", query="exportaciones"), llamada("submit_answer", respuesta="x", valor=None, fecha=None)]
        return textos[len(vistos) - 1], 10, 5, 0.0

    episodio("pregunta", "SISTEMA", generar, herramientas)
    assert vistos[0][0] == {"role": "system", "content": "SISTEMA"}
    assert json.loads(vistos[1][-1]["content"])[0]["serie_id"] == "ser_exportaciones"


def test_sin_llamada_corta_el_episodio(herramientas):
    log, _ = episodio("p", None, guion("Creo que la respuesta es 5."), herramientas)
    assert log["sin_llamada"] and log["submit"] is None and log["pasos"] == 1


def test_json_invalido_corta_el_episodio(herramientas):
    log, _ = episodio("p", None, guion("<tool_call>{roto}</tool_call>"), herramientas)
    assert log["formato_invalido"] and not log["sin_llamada"]


def test_contexto_excedido(herramientas):
    log, _ = episodio("p", None, lambda m: (None, 9999, 0, 0.0), herramientas)
    assert log["contexto_excedido"] and log["submit"] is None


def test_herramienta_desconocida_y_query_no_textual_devuelven_error_al_modelo(herramientas):
    generar = guion(llamada("borrar_todo"), llamada("search_series", query=123), llamada("submit_answer", valor=None, fecha=None))
    log, msgs = episodio("p", None, generar, herramientas)
    herramienta = [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]
    assert all(h["status"] == 400 for h in herramienta) and log["submit"] is not None


def test_maximo_de_pasos(herramientas):
    log, _ = episodio("p", None, lambda m: (llamada("search_series", query="ipc"), 10, 5, 0.0), herramientas, max_pasos=3)
    assert log["pasos"] == 3 and log["submit"] is None and log["n_search"] == 3


def test_ids_ultima_es_la_ultima_consulta(herramientas):
    generar = guion(llamada("get_series", ids="ser_emae"), llamada("get_series", ids="ser_pbi_anual"),
                    llamada("submit_answer", valor=None, fecha=None))
    log, _ = episodio("p", None, generar, herramientas)
    assert log["ids_ultima"] == "ser_pbi_anual" and log["n_get"] == 2

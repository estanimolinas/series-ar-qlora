import json

from conftest import ApiFalsa
from series_qlora.herramientas import (
    Herramientas,
    clave,
    construir_indice,
    consultar_api,
    crear_cache,
    textos_indice,
)


def ids(resultado):
    return [x["serie_id"] for x in json.loads(resultado)]


def test_clave_es_canonica():
    assert clave({"b": 2, "a": "x", "c": None}) == clave({"a": "x", "b": "2"})


def test_busqueda_encuentra_por_titulo_y_prefijo(herramientas):
    assert set(ids(herramientas.search_series("precios consumidor"))[:2]) == {"ser_ipc_nivel", "ser_ipc_alim"}
    # prefijo de 5 letras: "exportacion" encuentra "exportaciones"
    assert ids(herramientas.search_series("exportacion"))[0] == "ser_exportaciones"


def test_busqueda_desempata_por_orden_del_catalogo(tmp_path):
    import pandas as pd

    orden = ["z_serie", "a_serie", "m_serie", "b_serie", "y_serie", "c_serie"]
    cat = pd.DataFrame({"serie_id": orden, "serie_titulo": "Tráfico telefónico", "distribucion_titulo": "Telefonía",
                        "frecuencia": "mensual", "serie_unidades": None, "dataset_titulo": "Telefonía",
                        "dataset_tema": "Servicios", "serie_descripcion": None}).set_index("serie_id")
    construir_indice(tmp_path / "i.db", cat)
    crear_cache(tmp_path / "c.db")
    h = Herramientas(tmp_path / "c.db", tmp_path / "i.db", cat)
    # seis series idénticas: el resultado sigue el orden del catálogo, y con k menor corta en ese orden
    assert ids(h.search_series("tráfico telefónico")) == orden
    assert ids(h.search_series("tráfico telefónico", k=3)) == orden[:3]


def test_busqueda_prefiere_el_titulo_sobre_la_distribucion(herramientas):
    # "mensual" aparece solo en la distribución de las series IPC; "actividad" en el título de ser_emae
    assert ids(herramientas.search_series("actividad"))[0] == "ser_emae"


def test_textos_indice_arma_titulo_y_contexto(catalogo):
    t = textos_indice(catalogo)
    assert t.loc["ser_emae", "titulo"] == "actividad economica, nivel general"
    assert t.loc["ser_emae", "contexto"] == (
        "emae medicion mensual de la actividad estimador de actividad actividad indice mensual")
    # campos vacíos (None) no dejan "none" ni "nan" en el texto
    assert "none" not in t.loc["ser_pbi_anual", "contexto"] and "nan" not in t.loc["ser_pbi_anual", "contexto"]


def test_busqueda_encuentra_por_el_contexto_aunque_el_titulo_no_lo_diga(herramientas):
    # "sector externo" es solo el tema del dataset de exportaciones; "mensual" no está en ningún título
    assert ids(herramientas.search_series("sector externo"))[0] == "ser_exportaciones"
    assert "ser_emae" in ids(herramientas.search_series("mensual"))


def test_busqueda_sin_tokens_utiles_devuelve_lista_vacia(herramientas):
    assert herramientas.search_series("de la el") == "[]"
    assert herramientas.search_series("") == "[]"


def test_busqueda_no_se_rompe_con_caracteres_especiales(herramientas):
    assert isinstance(json.loads(herramientas.search_series('precios" OR (*) NEAR')), list)


def test_busqueda_devuelve_metadatos(herramientas):
    r = json.loads(herramientas.search_series("producto interno"))[0]
    assert r == {"serie_id": "ser_pbi_anual", "titulo": "Producto interno bruto a precios constantes",
                 "distribucion": "PIB anual", "frecuencia": "anual"}


def test_get_series_cachea_y_no_repite_la_llamada(herramientas, api):
    args = {"ids": "ser_emae", "start_date": "2020-01", "end_date": "2020-01"}
    a = json.loads(herramientas.get_series(args))
    b = json.loads(herramientas.get_series(dict(reversed(list(args.items())))))   # mismo pedido, otro orden
    assert a == b and a["status"] == 200 and a["serie_id"] == "ser_emae"
    assert len(api.llamadas) == 1 and herramientas.fallos_cache == 1


def test_get_series_agrega_metadatos_del_catalogo(herramientas):
    r = json.loads(herramientas.get_series({"ids": "ser_pbi_anual"}))
    assert r["frecuencia"] == "anual" and r["unidades"] is None and r["data"]


def test_get_series_rechaza_parametros_no_permitidos_sin_llamar_a_la_api(herramientas, api):
    r = json.loads(herramientas.get_series({"ids": "ser_emae", "format": "csv"}))
    assert r["status"] == 400 and "format" in r["error"]
    assert json.loads(herramientas.get_series({"limit": 1}))["status"] == 400
    assert api.llamadas == []


def test_get_series_guarda_los_400_y_los_devuelve_como_error(tmp_path, catalogo):
    crear_cache(tmp_path / "c.db")
    construir_indice(tmp_path / "i.db", catalogo)
    api = ApiFalsa(respuestas={"ser_emae": {"errors": [{"error": "fecha inválida"}]}}, status=400)
    h = Herramientas(tmp_path / "c.db", tmp_path / "i.db", catalogo, api=api)
    r = json.loads(h.get_series({"ids": "ser_emae", "start_date": "malo"}))
    assert r["status"] == 400 and r["error"] == [{"error": "fecha inválida"}]
    h.get_series({"ids": "ser_emae", "start_date": "malo"})
    assert len(api.llamadas) == 1          # el 400 también se cachea
    h.cerrar()


def test_get_series_sin_red_no_cachea_el_fallo(tmp_path, catalogo):
    crear_cache(tmp_path / "c.db")
    construir_indice(tmp_path / "i.db", catalogo)
    h = Herramientas(tmp_path / "c.db", tmp_path / "i.db", catalogo, api=lambda p: (None, None))
    r = json.loads(h.get_series({"ids": "ser_emae"}))
    assert r["status"] is None and h.sin_red == 1
    assert h.cache.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 0
    h.cerrar()


class Resp:
    def __init__(self, status, cuerpo=None):
        self.status_code, self._c = status, cuerpo

    def json(self):
        return self._c


def test_consultar_api_reintenta_y_acepta_200():
    respuestas = iter([Resp(500), RuntimeError("timeout"), Resp(200, {"data": []})])
    esperas = []

    def get(*a, **k):
        r = next(respuestas)
        if isinstance(r, Exception):
            raise r
        return r

    assert consultar_api({"ids": "x"}, get=get, dormir=esperas.append) == (200, {"data": []})
    assert esperas == [1, 2]


def test_consultar_api_se_rinde_tras_los_intentos():
    esperas = []
    assert consultar_api({"ids": "x"}, intentos=3, get=lambda *a, **k: Resp(503), dormir=esperas.append) == (None, None)
    assert len(esperas) == 3

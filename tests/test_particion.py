import numpy as np
import pandas as pd
import pytest

from series_qlora import config
from series_qlora.particion import Reparto, datasets_de, familias_de, particionar


def datasets(titulos):
    return pd.DataFrame({"titulo": titulos}, index=pd.Index([str(i) for i in range(len(titulos))], name="dataset_id"))


def test_familias_une_titulos_casi_iguales():
    f = familias_de(datasets(["Exportaciones por rubro, versión 1", "Exportaciones por rubro, versión 2", "Salarios"]),
                    uniones=())
    assert f["0"] == f["1"] and f["2"] != f["0"]


def test_familias_une_por_primera_frase():
    f = familias_de(datasets(["Precios mayoristas. Nivel general", "Precios mayoristas. Por rubro y destino final del producto",
                              "Reservas"]), uniones=())
    assert f["0"] == f["1"] and f["2"] != f["0"]


def test_familias_une_manualmente_y_tolera_ids_inexistentes():
    d = datasets(["aaaaaaaaaa bbbbbbbbbb", "cccccccccc dddddddddd", "eeeeeeeeee ffffffffff"])
    f = familias_de(d, uniones=(("0", "2", "999"),))
    assert f["0"] == f["2"] and f["1"] != f["0"]


def test_familias_ignora_tildes_y_mayusculas():
    f = familias_de(datasets(["Índice de Precios", "indice de precios"]), uniones=())
    assert f["0"] == f["1"]


@pytest.fixture
def catalogo_sintetico():
    rng = np.random.RandomState(0)
    letras = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    temas = ["Precios", "Actividad", "Sector Externo"]
    filas, n_ds = [], 0

    def agregar(titulo, tema, n_series):
        nonlocal n_ds
        for j in range(n_series):
            filas.append({"serie_id": f"s{n_ds}_{j}", "dataset_id": str(n_ds), "dataset_titulo": titulo,
                          "dataset_tema": tema, "frecuencia": rng.choice(["mensual", "anual"]),
                          "estado": rng.choice(["normal", "normal", "normal", "sin_datos"])})
        n_ds += 1

    for _ in range(70):                                    # datasets independientes
        agregar("".join(rng.choice(letras, 18)), temas[rng.randint(3)], rng.randint(5, 30))
    for k in range(4):                                     # una familia de 4 tablas
        agregar(f"Tabla regional del indice de precios. Parte {k}", "Precios", 20)
    return pd.DataFrame(filas)


def test_ninguna_familia_cruza_splits_y_todo_dataset_tiene_split(catalogo_sintetico):
    r = particionar(catalogo_sintetico, n_semillas=50)["df"]
    assert r["split"].isin(config.SPLITS).all()
    assert (r.groupby("familia")["split"].nunique() == 1).all()
    assert r["familia"].nunique() < r["dataset_id"].nunique()          # la familia de 4 tablas se agrupó


def test_es_determinista(catalogo_sintetico):
    a, b = particionar(catalogo_sintetico, n_semillas=30), particionar(catalogo_sintetico, n_semillas=30)
    assert a["semilla"] == b["semilla"] and a["puntaje"] == b["puntaje"]
    pd.testing.assert_series_equal(a["df"]["split"], b["df"]["split"])


def test_reparto_cercano_a_80_10_10(catalogo_sintetico):
    u = particionar(catalogo_sintetico, n_semillas=200)["usables"]
    total = sum(u.values())
    assert 0.65 < u["train"] / total < 0.92 and u["val"] > 0 and u["test"] > 0


def test_el_puntaje_devuelto_es_el_de_la_semilla_elegida(catalogo_sintetico):
    res = particionar(catalogo_sintetico, n_semillas=40)
    df = catalogo_sintetico.copy()
    df["dataset_id"] = df["dataset_id"].astype(str)
    ds = datasets_de(df)
    ds["familia"] = familias_de(ds)
    df["familia"] = df["dataset_id"].map(ds["familia"])
    reparto = Reparto(df, ds)
    assert reparto.puntaje(reparto.asignar(res["semilla"])) == pytest.approx(res["puntaje"])


def test_mas_semillas_nunca_empeoran_el_puntaje(catalogo_sintetico):
    assert (particionar(catalogo_sintetico, n_semillas=100)["puntaje"]
            <= particionar(catalogo_sintetico, n_semillas=10)["puntaje"])

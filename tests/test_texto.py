from series_qlora.texto import consulta_de, norm


def test_norm_quita_tildes_guiones_bajos_y_espacios():
    assert norm("  Índice_de   Precios ") == "indice de precios"


def test_norm_tolera_none_y_nan():
    assert norm(None) == "" and norm(float("nan")) == ""


def test_consulta_descarta_fechas_meses_ordinales_y_muletillas():
    q = consulta_de("¿Cuál fue la variación interanual del IPC en marzo de 2020?")
    assert q == "IPC"


def test_consulta_conserva_palabras_de_contenido_y_su_forma_original():
    q = consulta_de("¿Cuál fue el valor del Índice de Precios al Consumidor en el primer trimestre de 2019?")
    assert q.split() == ["valor", "Índice", "Precios", "Consumidor", "trimestre"]


def test_consulta_elimina_repetidos_y_tokens_de_una_letra():
    assert consulta_de("exportaciones x exportaciones Exportaciones") == "exportaciones"

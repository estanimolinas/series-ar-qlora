"""Prueba de humo de las figuras con datos sintéticos: se generan en un directorio temporal y no se guardan."""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from series_qlora import graficos  # noqa: E402


@pytest.fixture
def resultados():
    rng = np.random.RandomState(0)
    n = 80
    tipo = np.array((["sin_datos", "ultimo_dato", "valor_puntual", "variacion_interanual"] * n)[:n])
    ok_b, ok_f = rng.rand(n) < 0.3, rng.rand(n) < 0.7
    causas = ["no_termino", "serie_equivocada", "valor_equivocado", "fecha_equivocada"]
    mk = lambda ok: pd.DataFrame({"spec_id": [f"x-{i}" for i in range(n)], "tipo": tipo, "correcta": ok,  # noqa: E731
                                  "causa": np.where(ok, "correcta", rng.choice(causas, n))})
    return mk(ok_b), mk(ok_f)


def test_wilson_recorta_a_cero_y_uno():
    assert graficos.wilson(0, 10)[0] == 0.0 and graficos.wilson(10, 10)[1] == 1.0


def test_formato_con_coma():
    assert graficos._es(72.84, 1) == "72,8" and graficos._es_signo(-1.0) == "−1,0"


def test_bootstrap_de_familias_es_reproducible(resultados):
    A, B = resultados
    fam = {s: i % 5 for i, s in enumerate(A["spec_id"])}
    a = graficos.bootstrap_familias(A, B, fam, n_boot=200)
    assert a == graficos.bootstrap_familias(A, B, fam, n_boot=200)
    assert a[1] <= a[0] <= a[2] and a[3] == 5


def test_se_generan_las_cinco_figuras_en_png_y_pdf(resultados, tmp_path, monkeypatch):
    A, B = resultados
    graficos.aplicar_estilo()
    monkeypatch.setattr(graficos.plt, "show", lambda *a, **k: None)
    graficos.fig_exactitud_por_tipo(A, B, str(tmp_path / "f1.png"))
    graficos.fig_causas(A, B, str(tmp_path / "f2.png"))
    graficos.fig_validacion_vs_test(A, B, A, B, str(tmp_path / "f3.png"))
    graficos.fig_perdida(str(tmp_path / "f4.png"))
    V = graficos.fig_emparejado(A, B, str(tmp_path / "f5.png"))
    for i in range(1, 6):
        assert (tmp_path / f"f{i}.png").stat().st_size > 0 and (tmp_path / f"f{i}.pdf").stat().st_size > 0
    assert V.sum() == len(A)
    assert "Figura 5" in graficos.pies_de_figura(A, B, A, B, V, None)

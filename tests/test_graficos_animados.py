import re
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from series_qlora.graficos import PERDIDA_CONTROL, PERDIDA_ENTRENAMIENTO, punto
from series_qlora.graficos_animados import generar_animadas, svg_exactitud_por_tipo, svg_perdida

TIPOS = ["sin_datos", "ultimo_dato", "valor_puntual", "variacion_interanual"]


def corrida(aciertos_por_tipo, n=20):
    filas = []
    for t, k in aciertos_por_tipo.items():
        filas += [{"tipo": t, "correcta": i < k} for i in range(n)]
    return pd.DataFrame(filas)


@pytest.fixture
def pares():
    return (corrida({"sin_datos": 2, "ultimo_dato": 4, "valor_puntual": 10, "variacion_interanual": 6}),
            corrida({"sin_datos": 16, "ultimo_dato": 14, "valor_puntual": 14, "variacion_interanual": 12}))


def test_los_svg_son_xml_valido_sin_scripts_ni_recursos_externos(pares):
    for s in (svg_exactitud_por_tipo(*pares), svg_perdida()):
        raiz = ET.fromstring(s)
        assert raiz.tag.endswith("svg")
        assert "<script" not in s and "http://" not in s.replace("http://www.w3.org/2000/svg", "")
        assert "prefers-reduced-motion" in s and "prefers-color-scheme: dark" in s


def test_las_barras_muestran_los_mismos_valores_que_las_estaticas(pares):
    A, B = pares
    s = svg_exactitud_por_tipo(A, B)
    for df_ in (A, B):
        total = punto(df_)[0]
        assert f"{total:.1f}".replace(".", ",") in s
        for t in TIPOS:
            assert f"{punto(df_[df_['tipo'] == t])[0]:.1f}".replace(".", ",") in s
    assert "n = 80" in s and s.count("n = 20") == 4


def test_la_altura_de_cada_barra_es_proporcional_al_valor(pares):
    A, B = pares
    raiz = ET.fromstring(svg_exactitud_por_tipo(A, B))
    ns = "{http://www.w3.org/2000/svg}"
    barras = [r for r in raiz.iter(ns + "rect") if r.get("class", "").startswith("b")]
    assert len(barras) == 10
    alturas = [float(b.get("height")) for b in barras]
    esperados = [punto(df_ if g is None else df_[df_["tipo"] == g])[0] for g in [None] + TIPOS for df_ in (A, B)]
    por_ciento = 3.3                      # 330 px de alto para 100 %
    assert all(h == pytest.approx(v * por_ciento, abs=0.06) for h, v in zip(alturas, esperados, strict=True))


def test_la_curva_tiene_todos_los_valores_del_registro():
    s = svg_perdida()
    assert s.count('r="5.5"') >= len(PERDIDA_ENTRENAMIENTO)
    desc = re.search(r"<desc[^>]*>(.*?)</desc>", s).group(1)
    for v in PERDIDA_ENTRENAMIENTO:
        assert f"{v:.4f}".replace(".", ",") in desc
    for paso, v in PERDIDA_CONTROL.items():
        assert f"{paso}: {v:.4f}".replace(".", ",") in desc


def test_los_nombres_de_animacion_no_se_repiten(pares):
    for s in (svg_exactitud_por_tipo(*pares), svg_perdida()):
        nombres = re.findall(r"@keyframes (\w+)", s)
        assert len(nombres) == len(set(nombres))
        usados = set(re.findall(r"animation:(\w+) ", s))
        assert usados - {"none"} <= set(nombres)


def test_generar_animadas_escribe_dos_archivos(pares, tmp_path):
    rutas = generar_animadas(*pares, str(tmp_path / "figs"))
    assert [r.rsplit("/", 1)[1] for r in rutas] == ["fig1_exactitud_por_tipo_animada.svg",
                                                     "fig4_curva_de_perdida_animada.svg"]
    assert all((tmp_path / "figs" / r.rsplit("/", 1)[1]).stat().st_size > 1000 for r in rutas)

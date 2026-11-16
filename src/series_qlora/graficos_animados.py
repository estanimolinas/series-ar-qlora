"""Versiones animadas (SVG) de dos figuras: la comparación por tipo y la curva de pérdida.

Usan exactamente los mismos números que las figuras estáticas (`graficos.py`): los intervalos son de Wilson y la
curva sale de los valores del registro de entrenamiento. La animación solo cambia cómo aparecen; los valores
finales son los medidos. Son SVG con CSS, sin JavaScript, así que GitHub los anima dentro del README. Con
`prefers-reduced-motion` se ven ya completos. Las estáticas (PNG/PDF) siguen siendo las que se citan.
"""
from __future__ import annotations

import math
import os
from xml.sax.saxutils import escape

from .graficos import (
    C_BASE,
    C_FT,
    N_BASE,
    N_FT,
    NOMBRE_TIPO,
    PASOS,
    PERDIDA_CONTROL,
    PERDIDA_ENTRENAMIENTO,
    TIPOS,
    _es,
    punto,
)

CICLO = 14.0           # segundos que dura una vuelta; el gráfico completo queda quieto casi todo el ciclo
FUENTE = '-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif'


def _estilo(extra: str) -> str:
    return f"""<style>
:root{{--fondo:#ffffff;--texto:#1f2328;--suave:#59636e;--grilla:#e3e6ea;--tinta:#222222;--base:{C_BASE};--ft:{C_FT}}}
@media (prefers-color-scheme: dark){{:root{{--fondo:#0d1117;--texto:#e6edf3;--suave:#9198a1;--grilla:#2b323a;--tinta:#d0d7de;--base:#b48ad6;--ft:#4cc38a}}}}
text{{font-family:{FUENTE};fill:var(--texto)}} .suave{{fill:var(--suave)}}
.grilla{{stroke:var(--grilla);stroke-width:1}} .eje{{stroke:var(--suave);stroke-width:1}}
{extra}
</style>"""


def _pct(t: float) -> str:
    return f"{t / CICLO * 100:.2f}%"


def _kf_aparece(nombre: str, t0: float, t1: float, desde: str, hasta: str) -> str:
    """Estado `desde` hasta t0, transición a `hasta` en t1 y se mantiene; vuelve a `desde` al reiniciar el ciclo."""
    return (f"@keyframes {nombre}{{0%,{_pct(t0)}{{{desde}}}{_pct(t1)},100%{{{hasta}}}}}")


# La figura entera se desvanece al final del ciclo para que el reinicio no se note
KF_CICLO = (f"@keyframes ciclo{{0%{{opacity:0}}{_pct(0.3)}{{opacity:1}}{_pct(CICLO - 1.4)}{{opacity:1}}"
            f"{_pct(CICLO - 0.5)},100%{{opacity:0}}}} .ciclo{{animation:ciclo {CICLO}s infinite}}")
REDUCIDO = "@media (prefers-reduced-motion: reduce){*{animation:none !important;opacity:1 !important;transform:none !important}}"


def _lineas_texto(x: float, y: float, texto: str, clase: str, ancho_linea: int = 16, **attrs) -> str:
    extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    partes = texto.split("\n")
    tsp = "".join(f'<tspan x="{x}" dy="{0 if i == 0 else ancho_linea}">{escape(p)}</tspan>' for i, p in enumerate(partes))
    return f'<text x="{x}" y="{y}" class="{clase}" {extra}>{tsp}</text>'


def svg_exactitud_por_tipo(A, B) -> str:
    """Barras con intervalos de Wilson 95 %: total y cuatro tipos de pregunta, modelo base contra ajustado."""
    W, H = 900, 500
    x0, x1, y_base, y_top = 70, 880, 400, 70        # área del gráfico; y_base = 0 %, y_top = 100 %
    grupos = [None] + TIPOS
    etiquetas = [f"Total\nn = {len(A)}"] + [f"{NOMBRE_TIPO[t].replace(chr(10), ' ')}\nn = {int((A['tipo'] == t).sum())}"
                                           for t in TIPOS]
    ancho_g = (x1 - x0) / len(grupos)
    w = 46
    py = lambda v: y_base - v / 100 * (y_base - y_top)  # noqa: E731
    css, cuerpo, desc = [], [], []
    for gi, g in enumerate(grupos):
        cx = x0 + ancho_g * (gi + 0.5)
        for si, (df_, color, nombre) in enumerate(((A, "base", N_BASE), (B, "ft", N_FT))):
            p, lo, hi = punto(df_ if g is None else df_[df_["tipo"] == g])
            i = gi * 2 + si
            t0 = 0.5 + gi * 0.35 + si * 0.15
            bx = cx + (-w - 2 if si == 0 else 2)
            h = p / 100 * (y_base - y_top)
            css.append(_kf_aparece(f"b{i}", t0, t0 + 1.0, "transform:scaleY(0)", "transform:scaleY(1)"))
            css.append(_kf_aparece(f"e{i}", t0 + 0.9, t0 + 1.4, "opacity:0", "opacity:1"))
            css.append(f".b{i}{{transform-box:fill-box;transform-origin:50% 100%;animation:b{i} {CICLO}s ease-out infinite}}"
                       f".e{i}{{animation:e{i} {CICLO}s infinite}}")
            xc = bx + w / 2
            cuerpo.append(f'<rect class="b{i}" x="{bx:.1f}" y="{y_base - h:.1f}" width="{w}" height="{h:.1f}" rx="2" fill="var(--{color})"/>')
            cuerpo.append(
                f'<g class="e{i}"><line x1="{xc:.1f}" x2="{xc:.1f}" y1="{py(lo):.1f}" y2="{py(hi):.1f}" stroke="var(--tinta)" stroke-width="1.5"/>'
                f'<line x1="{xc - 5:.1f}" x2="{xc + 5:.1f}" y1="{py(lo):.1f}" y2="{py(lo):.1f}" stroke="var(--tinta)" stroke-width="1.5"/>'
                f'<line x1="{xc - 5:.1f}" x2="{xc + 5:.1f}" y1="{py(hi):.1f}" y2="{py(hi):.1f}" stroke="var(--tinta)" stroke-width="1.5"/>'
                f'<text x="{xc:.1f}" y="{py(hi) - 7:.1f}" text-anchor="middle" font-size="14">{_es(p, 1)}</text></g>')
            desc.append(f"{'total' if g is None else g} {nombre}: {_es(p, 1)} % (IC 95 % {_es(lo, 1)} a {_es(hi, 1)})")
    grilla = "".join(f'<line class="grilla" x1="{x0}" x2="{x1}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>'
                     f'<text class="suave" x="{x0 - 10}" y="{py(v) + 4:.1f}" text-anchor="end" font-size="13">{v}</text>'
                     for v in range(0, 101, 20))
    xt = "".join(_lineas_texto(round(x0 + ancho_g * (gi + 0.5), 1), y_base + 24, e, "", 17, text_anchor="middle", font_size="14")
                 for gi, e in enumerate(etiquetas))
    leyenda = (f'<rect x="{x0}" y="22" width="14" height="14" rx="2" fill="var(--base)"/>'
               f'<text x="{x0 + 22}" y="34" font-size="14">{escape(N_BASE)}</text>'
               f'<rect x="{x0 + 300}" y="22" width="14" height="14" rx="2" fill="var(--ft)"/>'
               f'<text x="{x0 + 322}" y="34" font-size="14">{escape(N_FT)}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">\n'
            f'<title id="t">Exactitud por tipo de pregunta en test: modelo base contra QLoRA</title>\n'
            f'<desc id="d">{escape("; ".join(desc))}. Barras de error: intervalo de Wilson al 95 %.</desc>\n'
            f'{_estilo(KF_CICLO + chr(10) + chr(10).join(css) + chr(10) + REDUCIDO)}\n'
            f'<rect width="{W}" height="{H}" fill="var(--fondo)"/>{leyenda}{grilla}'
            f'<text class="suave" x="22" y="{(y_base + y_top) / 2}" font-size="13" text-anchor="middle" '
            f'transform="rotate(-90 22 {(y_base + y_top) / 2})">Respuestas correctas (%)</text>'
            f'<line class="eje" x1="{x0}" x2="{x1}" y1="{y_base}" y2="{y_base}"/>{xt}'
            f'<g class="ciclo">{"".join(cuerpo)}</g></svg>\n')


def svg_perdida() -> str:
    """Curva de pérdida (escala logarítmica) que se dibuja paso a paso, con los puntos de control."""
    W, H = 760, 500
    x0, x1, y0, y1 = 80, 730, 410, 60                # y0 = 0,05 ; y1 = 1,2 (log)
    lmin, lmax = math.log10(0.05), math.log10(1.2)
    px = lambda s: x0 + s / 105 * (x1 - x0)  # noqa: E731
    py = lambda v: y0 - (math.log10(v) - lmin) / (lmax - lmin) * (y0 - y1)  # noqa: E731
    pts = [(px(s), py(v)) for s, v in zip(PASOS, PERDIDA_ENTRENAMIENTO, strict=True)]
    acum = [0.0]
    for (xa, ya), (xb, yb) in zip(pts, pts[1:], strict=False):
        acum.append(acum[-1] + math.hypot(xb - xa, yb - ya))
    total = acum[-1]
    T0, T1 = 0.8, 8.0                                # la curva se dibuja entre T0 y T1
    tt = [T0 + (T1 - T0) * a / total for a in acum]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    css = [_kf_aparece("trazo", T0, T1, "stroke-dashoffset:1", "stroke-dashoffset:0"),
           ".trazo{stroke-dasharray:1;stroke-dashoffset:1;animation:trazo " + f"{CICLO}s linear infinite}}"]
    marcas = []
    for i, ((x, y), t) in enumerate(zip(pts, tt, strict=True)):
        css.append(_kf_aparece(f"m{i}", t - 0.05, t + 0.2, "opacity:0", "opacity:1"))
        css.append(f".m{i}{{animation:m{i} {CICLO}s infinite}}")
        marcas.append(f'<circle class="m{i}" cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="var(--ft)"/>')
    for j, (paso, v) in enumerate(PERDIDA_CONTROL.items()):
        t = tt[PASOS.index(paso)] + 0.4
        css.append(_kf_aparece(f"c{j}", t, t + 0.3, "opacity:0", "opacity:1"))
        css.append(f".c{j}{{animation:c{j} {CICLO}s infinite}}")
        marcas.append(f'<rect class="c{j}" x="{px(paso) - 6.5:.1f}" y="{py(v) - 6.5:.1f}" width="13" height="13" fill="var(--base)"/>'
                      f'<text class="c{j}" x="{px(paso) + 12:.1f}" y="{py(v) + 5:.1f}" font-size="14">{_es(v, 3)}</text>')
    t_fin = tt[-1] + 0.3
    css.append(_kf_aparece("etq", t_fin, t_fin + 0.4, "opacity:0", "opacity:1"))
    css.append(f".etq{{animation:etq {CICLO}s infinite}}")
    etq = (f'<text class="etq" x="{pts[0][0] + 12:.1f}" y="{pts[0][1] + 5:.1f}" font-size="14">{_es(PERDIDA_ENTRENAMIENTO[0], 3)}</text>'
           f'<text class="etq" x="{pts[-1][0] - 8:.1f}" y="{pts[-1][1] + 26:.1f}" font-size="14" text-anchor="end">'
           f'{_es(PERDIDA_ENTRENAMIENTO[-1], 3)}</text>')
    grilla = "".join(f'<line class="grilla" x1="{x0}" x2="{x1}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>'
                     f'<text class="suave" x="{x0 - 10}" y="{py(v) + 4:.1f}" text-anchor="end" font-size="13">{_es(v, 1)}</text>'
                     for v in (0.1, 0.2, 0.5, 1.0))
    xt = "".join(f'<text class="suave" x="{px(s):.1f}" y="{y0 + 22}" text-anchor="middle" font-size="13">{s}</text>'
                 for s in (0, 20, 40, 60, 80, 100))
    leyenda = (f'<circle cx="{x0 + 7}" cy="29" r="5.5" fill="var(--ft)"/><text x="{x0 + 20}" y="34" font-size="14">Entrenamiento</text>'
               f'<rect x="{x0 + 160}" y="22" width="13" height="13" fill="var(--base)"/>'
               f'<text x="{x0 + 181}" y="34" font-size="14">Control (3 familias apartadas)</text>')
    desc = ("Pérdida de entrenamiento por paso: " + ", ".join(f"{s}: {_es(v, 4)}" for s, v in zip(PASOS, PERDIDA_ENTRENAMIENTO, strict=True))
            + ". Control: " + ", ".join(f"{s}: {_es(v, 4)}" for s, v in PERDIDA_CONTROL.items()) + ".")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">\n'
            f'<title id="t">Curva de pérdida del entrenamiento QLoRA</title><desc id="d">{escape(desc)}</desc>\n'
            f'{_estilo(KF_CICLO + chr(10) + chr(10).join(css) + chr(10) + REDUCIDO)}\n'
            f'<rect width="{W}" height="{H}" fill="var(--fondo)"/>{leyenda}{grilla}'
            f'<line class="eje" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}"/>{xt}'
            f'<text class="suave" x="{(x0 + x1) / 2}" y="{y0 + 48}" text-anchor="middle" font-size="13">Paso de entrenamiento</text>'
            f'<text class="suave" x="24" y="{(y0 + y1) / 2}" text-anchor="middle" font-size="13" '
            f'transform="rotate(-90 24 {(y0 + y1) / 2})">Pérdida (escala logarítmica)</text>'
            f'<g class="ciclo"><path class="trazo" pathLength="1" d="{d}" fill="none" stroke="var(--ft)" stroke-width="2.6" '
            f'stroke-linejoin="round" stroke-linecap="round"/>{"".join(marcas)}{etq}</g></svg>\n')


def generar_animadas(A_test, B_test, destino: str) -> list[str]:
    """Escribe las dos figuras animadas en `destino` y devuelve las rutas."""
    os.makedirs(destino, exist_ok=True)
    salidas = {"fig1_exactitud_por_tipo_animada.svg": svg_exactitud_por_tipo(A_test, B_test),
               "fig4_curva_de_perdida_animada.svg": svg_perdida()}
    rutas = []
    for nombre, contenido in salidas.items():
        ruta = os.path.join(destino, nombre)
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(contenido)
        rutas.append(ruta)
    return rutas

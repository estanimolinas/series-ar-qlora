import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_BASE, C_FT = "#7C4DA0", "#2E8B57"
TINTA = "#222222"
N_BASE, N_FT = "Base + prompt optimizado", "QLoRA, sin prompt de sistema"
TIPOS = ["sin_datos", "ultimo_dato", "valor_puntual", "variacion_interanual"]
NOMBRE_TIPO = {"sin_datos": "Sin datos", "ultimo_dato": "Último dato", "valor_puntual": "Valor puntual",
               "variacion_interanual": "Variación\ninteranual"}
NOMBRE_CAUSA = {"no_termino": "Sin respuesta final\n(sin submit_answer)", "serie_equivocada": "Serie equivocada",
                "valor_equivocado": "Valor equivocado", "fecha_equivocada": "Fecha equivocada",
                "convencion_porcentaje": "Fracción en vez\nde porcentaje"}

# Valores impresos por el entrenamiento v1; se copian del registro, no hay archivo de donde leerlos.
PASOS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PERDIDA_ENTRENAMIENTO = [1.0967, 0.3605, 0.2072, 0.154, 0.1269, 0.1128, 0.0953, 0.0945, 0.0894, 0.0866]
PERDIDA_CONTROL = {50: 0.1286, 100: 0.1003}


def _es(x, dec=0):
    return f"{x:.{dec}f}".replace(".", ",")


def _es_signo(x, dec=1):
    return ("+" if x >= 0 else "−") + _es(abs(x), dec)


def _cargar_fuente():
    import urllib.request

    from matplotlib import font_manager
    carpeta = os.path.join(os.path.expanduser("~"), ".cache", "series_qlora_fuentes")
    try:
        os.makedirs(carpeta, exist_ok=True)
        for archivo in ("Lato-Regular.ttf", "Lato-Bold.ttf"):
            destino = os.path.join(carpeta, archivo)
            if not os.path.exists(destino):
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/google/fonts/main/ofl/lato/" + archivo, destino)
            font_manager.fontManager.addfont(destino)
        return "Lato"
    except Exception:  # sin red: fuente por defecto
        return "DejaVu Sans"


def aplicar_estilo():
    plt.rcParams.update({
        "font.family": _cargar_fuente(), "font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5, "legend.fontsize": 12.5, "text.color": TINTA, "axes.labelcolor": TINTA,
        "axes.linewidth": 1.0, "axes.edgecolor": TINTA, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 4.5,
        "ytick.major.size": 4.5, "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.color": TINTA,
        "ytick.color": TINTA, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def wilson(k, n, z=1.96):
    # Recortado a [0, 1]: en k = 0 el redondeo da -2.8e-17.
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def punto(d):
    k, n = int(d["correcta"].sum()), len(d)
    lo, hi = wilson(k, n)
    return k / n * 100, lo * 100, hi * 100


def bootstrap_familias(A, B, familias, n_boot=10000, semilla=0):
    # Remuestrea familias enteras: las preguntas de una misma familia no son independientes.
    M = A[["spec_id", "correcta"]].merge(B[["spec_id", "correcta"]], on="spec_id", suffixes=("_a", "_b"))
    M["familia"] = M["spec_id"].map(familias)
    diff = M["correcta_b"].astype(float) - M["correcta_a"].astype(float)
    g = diff.groupby(M["familia"]).agg(["sum", "count"])
    rng = np.random.RandomState(semilla)
    est = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(g), len(g))
        est.append(g["sum"].values[idx].sum() / g["count"].values[idx].sum())
    return diff.mean() * 100, np.percentile(est, 2.5) * 100, np.percentile(est, 97.5) * 100, len(g)


def _guardar(fig, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    base = os.path.splitext(ruta)[0]
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.05)
    plt.show()
    plt.close(fig)


def _barras(ax, x, valores, w, off, color, err=None, etiquetas=True, horizontal=False):
    kw = dict(color=color, linewidth=0, zorder=2)
    if horizontal:
        ax.barh(x + off, valores, height=w, **kw)
        return
    ax.bar(x + off, valores, w, **kw)
    if err is not None:
        ax.errorbar(x + off, valores, yerr=err, fmt="none", ecolor=TINTA, elinewidth=1.3,
                    capsize=4, capthick=1.3, zorder=3)
    if etiquetas:
        tope = valores + (err[1] if err is not None else 0)
        for xi, v, t in zip(x + off, valores, tope):
            ax.text(xi, t + 1.8, _es(v, 1), ha="center", va="bottom", fontsize=11.5)


def _leyenda_arriba(ax):
    from matplotlib.patches import Patch
    h = [Patch(facecolor=C_BASE, label=N_BASE), Patch(facecolor=C_FT, label=N_FT)]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, handlelength=1.4, columnspacing=2.2)


def fig_exactitud_por_tipo(A, B, ruta):
    grupos = [None] + TIPOS
    etiquetas = ["Total\n(n = %d)" % len(A)] + [
        "%s\n(n = %d)" % (NOMBRE_TIPO[t], int((A["tipo"] == t).sum())) for t in TIPOS]
    x = np.arange(len(grupos))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for off, df_, color in ((-w / 2 - 0.01, A, C_BASE), (w / 2 + 0.01, B, C_FT)):
        pts = [punto(df_ if g is None else df_[df_["tipo"] == g]) for g in grupos]
        p = np.array([q[0] for q in pts])
        err = [p - np.array([q[1] for q in pts]), np.array([q[2] for q in pts]) - p]
        _barras(ax, x, p, w, off, color, err)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Respuestas correctas (%)")
    _leyenda_arriba(ax)
    _guardar(fig, ruta)


def fig_causas(A, B, ruta):
    causas = [c for c in NOMBRE_CAUSA if (A["causa"] == c).any() or (B["causa"] == c).any()]
    causas.sort(key=lambda c: -((A["causa"] == c).sum() + (B["causa"] == c).sum()))
    y = np.arange(len(causas)) * 1.0
    h = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for off, df_, color in ((-h / 2 - 0.01, A, C_BASE), (h / 2 + 0.01, B, C_FT)):
        v = np.array([(df_["causa"] == c).sum() for c in causas])
        _barras(ax, y, v, h, off, color, horizontal=True)
        for yi, vi in zip(y + off, v):
            ax.text(vi + len(A) * 0.008, yi, str(vi), va="center", fontsize=11.5)
    ax.set_yticks(y)
    ax.set_yticklabels([NOMBRE_CAUSA[c] for c in causas])
    ax.invert_yaxis()
    ax.set_xlabel("Preguntas incorrectas (de %d)" % len(A))
    ax.set_xlim(0, ax.get_xlim()[1] * 1.05)
    _leyenda_arriba(ax)
    _guardar(fig, ruta)


def fig_validacion_vs_test(A_v, B_v, A_t, B_t, ruta):
    x = np.arange(2)
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 5.4))
    for off, (av, at), color in ((-w / 2 - 0.01, (A_v, A_t), C_BASE), (w / 2 + 0.01, (B_v, B_t), C_FT)):
        pts = [punto(av), punto(at)]
        p = np.array([q[0] for q in pts])
        err = [p - np.array([q[1] for q in pts]), np.array([q[2] for q in pts]) - p]
        _barras(ax, x, p, w, off, color, err)
    ax.set_xticks(x)
    ax.set_xticklabels(["Validación\n(n = %d)" % len(A_v), "Test\n(n = %d)" % len(A_t)])
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Respuestas correctas (%)")
    _leyenda_arriba(ax)
    _guardar(fig, ruta)


def fig_perdida(ruta):
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.plot(PASOS, PERDIDA_ENTRENAMIENTO, color=C_FT, lw=2.2, marker="o", markersize=7, label="Entrenamiento")
    ax.plot(list(PERDIDA_CONTROL), list(PERDIDA_CONTROL.values()), color=C_BASE, ls="none",
            marker="s", markersize=8, label="Control (3 familias apartadas)")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(["0,1", "0,2", "0,5", "1,0"])
    ax.minorticks_off()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Paso de entrenamiento")
    ax.set_ylabel("Pérdida (escala logarítmica)")
    ax.grid(True, color="#e3e3e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    _guardar(fig, ruta)


def fig_emparejado(A, B, ruta):
    M = A.merge(B, on="spec_id", suffixes=("_b", "_f"))
    n = len(M)
    cb, cf = M["correcta_b"].astype(bool), M["correcta_f"].astype(bool)
    V = np.array([[(cb & cf).sum(), (cb & ~cf).sum()], [(~cb & cf).sum(), (~cb & ~cf).sum()]])
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    from matplotlib.colors import LinearSegmentedColormap
    mapa = LinearSegmentedColormap.from_list("verde", ["#ffffff", "#b7dcc6", C_FT, "#12432a"])
    im = ax.imshow(V, cmap=mapa, vmin=0, vmax=V.max() * 1.15)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{V[i, j]}\n({_es(V[i, j] / n * 100, 1)} %)", ha="center", va="center", fontsize=15,
                    color="white" if V[i, j] > 0.5 * V.max() else TINTA)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Correcta", "Incorrecta"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Correcta", "Incorrecta"], rotation=90, va="center")
    ax.set_xlabel("QLoRA, sin prompt de sistema")
    ax.set_ylabel("Base + prompt optimizado")
    for s in ax.spines.values():
        s.set_visible(True)
    cb_ = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb_.set_label("Preguntas")
    cb_.outline.set_linewidth(1.0)
    _guardar(fig, ruta)
    return V


def pies_de_figura(A_t, B_t, A_v, B_v, V, dif):
    nt, nv = len(A_t), len(A_v)
    ci = (f"Diferencia total entre modelos: {_es_signo(dif[0])} puntos; IC del 95 % remuestreando las {dif[3]} "
          f"familias de datasets (10.000 réplicas): {_es_signo(dif[1])} a {_es_signo(dif[2])}.") if dif else ""
    return "\n\n".join([
        f"**Figura 1.** Porcentaje de respuestas correctas en test (n = {nt}), en total y por tipo de pregunta. "
        "Barras violeta: modelo base con prompt optimizado; barras verdes: modelo ajustado con QLoRA, sin prompt de sistema. "
        "Barras de error: intervalo de confianza del 95 % de Wilson, que supone preguntas independientes. " + ci,
        f"**Figura 2.** Causa de fallo de las preguntas incorrectas en test (n = {nt}). Cada pregunta incorrecta se asigna a "
        "su primera causa, en este orden: no terminó con `submit_answer`, serie equivocada, valor equivocado y fecha equivocada. "
        "Dentro de valor equivocado se cuenta aparte el caso en que el valor es correcto expresado como fracción en vez de porcentaje.",
        f"**Figura 3.** Porcentaje de respuestas correctas en validación (n = {nv}) y en test (n = {nt}). Barras de error: "
        "IC del 95 % de Wilson. Los niveles no son comparables entre conjuntos, porque contienen familias de datasets distintas.",
        "**Figura 4.** Pérdida durante el entrenamiento QLoRA (1 época, 100 pasos), copiada del registro de la corrida. "
        "La pérdida de entrenamiento se registra cada 10 pasos; la de control, solo en los pasos 50 y 100, sobre 3 familias de datasets apartadas.",
        f"**Figura 5.** Tabla de contingencia emparejada en test (n = {nt}): cada pregunta según si la respondió correctamente "
        f"el modelo base (filas) y el modelo ajustado (columnas). Ambas correctas: {V[0, 0]}; solo el base: {V[0, 1]}; "
        f"solo el ajustado: {V[1, 0]}; ninguna: {V[1, 1]}.",
    ])


def generar_todas(datos, destino, familias=None):
    aplicar_estilo()
    cargar = lambda s: (pd.read_csv(f"{datos}/baseline_{s}_tecnica_completa_optimizado.csv"),  # noqa: E731
                        pd.read_csv(f"{datos}/finetuned_{s}_tecnica_completa_basico.csv"))
    A_t, B_t = cargar("test")
    A_v, B_v = cargar("val")
    dif = None
    if familias is not None:
        dif = bootstrap_familias(A_t, B_t, familias)
        print(f"Diferencia (QLoRA - base): {dif[0]:+.1f} puntos | IC 95 % por familias: {dif[1]:+.1f} a {dif[2]:+.1f} "
              f"({dif[3]} familias)")
    fig_exactitud_por_tipo(A_t, B_t, f"{destino}/fig1_exactitud_por_tipo.png")
    fig_causas(A_t, B_t, f"{destino}/fig2_causas_de_fallo.png")
    fig_validacion_vs_test(A_v, B_v, A_t, B_t, f"{destino}/fig3_validacion_vs_test.png")
    fig_perdida(f"{destino}/fig4_curva_de_perdida.png")
    V = fig_emparejado(A_t, B_t, f"{destino}/fig5_emparejado_test.png")
    with open(f"{destino}/pies_de_figura.md", "w", encoding="utf-8") as f:
        f.write(pies_de_figura(A_t, B_t, A_v, B_v, V, dif) + "\n")
    print("Archivos en:", destino)
    print(sorted(os.listdir(destino)))

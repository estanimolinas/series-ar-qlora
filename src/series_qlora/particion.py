"""Partición train/val/test por familias de datasets.

Un dataset publicado en varias tablas (por provincia, por rubro) comparte estructura y títulos de series: si unas
tablas quedan en train y otras en test, el modelo vería en entrenamiento series casi idénticas a las de test. Por eso
se agrupan los datasets en familias y se reparte por familia entera.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

import numpy as np
import pandas as pd

from . import config


def _norm_titulo(s) -> str:
    # No usa texto.norm a propósito: esta es la normalización con la que se generó la partición publicada,
    # y cambiarla (p. ej. tratando "_" como espacio) podría alterar qué datasets se agrupan.
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.lower()).strip()


def datasets_de(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por dataset: título, tema más frecuente y cantidad de series usables."""
    usable = df["estado"].isin(config.ESTADOS_USABLES)
    return df.assign(usable=usable).groupby("dataset_id").agg(
        titulo=("dataset_titulo", "first"),
        tema=("dataset_tema", lambda s: s.mode().iloc[0]),
        usables=("usable", "sum"))


def familias_de(datasets: pd.DataFrame, umbral=config.UMBRAL_FAMILIA,
                uniones=config.UNIONES_MANUALES) -> pd.Series:
    """Familia de cada dataset (el id del dataset que hace de raíz del grupo), con tres reglas:

    a) títulos con similitud >= umbral; b) misma primera frase del título; c) uniones manuales.
    El resultado depende del orden de los ids (de ahí la raíz); `datasets` llega ordenado por dataset_id.
    """
    ids = datasets.index.tolist()
    tit = {d: _norm_titulo(datasets.loc[d, "titulo"]) for d in ids}
    padre = {d: d for d in ids}

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def unir(a, b):
        padre[raiz(b)] = raiz(a)

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if difflib.SequenceMatcher(None, tit[a], tit[b]).ratio() >= umbral:
                unir(a, b)

    por_frase: dict[str, list[str]] = {}
    for d in ids:
        por_frase.setdefault(tit[d].split(". ")[0], []).append(d)
    for lista in por_frase.values():
        for d in lista[1:]:
            unir(lista[0], d)

    for grupo in uniones:
        for d in grupo[1:]:
            if d in padre and grupo[0] in padre:
                unir(grupo[0], d)

    return pd.Series([raiz(d) for d in ids], index=datasets.index, name="familia")


class Reparto:
    """Matrices precomputadas por familia sobre las que se evalúa cada asignación candidata."""

    def __init__(self, df: pd.DataFrame, datasets: pd.DataFrame, tope=config.TOPE_FAMILIA):
        usable = df[df["estado"].isin(config.ESTADOS_USABLES)]
        self.familias = pd.Index(sorted(datasets["familia"].unique()))
        tam = usable.groupby("familia").size().reindex(self.familias).fillna(0)
        factor = np.minimum(1.0, tope / tam.clip(lower=1))
        tema_familia = datasets.groupby("familia")["tema"].agg(lambda s: s.mode().iloc[0]).reindex(self.familias)

        # Qué se quiere balancear entre splits: tema y frecuencia de las usables, y series sin datos o truncadas
        partes = [pd.crosstab(usable["familia"], usable[c]).add_prefix(c + "=") for c in ["dataset_tema", "frecuencia"]]
        estado = (pd.crosstab(df["familia"], df["estado"])
                  .reindex(columns=["sin_datos", "api_truncada"], fill_value=0).add_prefix("estado="))
        M = pd.concat([*partes, estado], axis=1).reindex(self.familias).fillna(0).mul(factor, axis=0)
        self.matriz, self.totales = M.values, M.sum().values
        self.peso = np.maximum(tam.values * factor.values, 1)
        self.tema = tema_familia.values
        self.tam_usables = tam.values
        self.p = np.array(config.PROPORCIONES)

    def asignar(self, semilla: int) -> np.ndarray:
        """Reparto voraz por tema: las familias, de mayor a menor peso (con ruido de la semilla), van al
        split con más déficit respecto de su proporción objetivo."""
        rng = np.random.RandomState(semilla)
        a = np.zeros(len(self.familias), dtype=int)
        for t in np.unique(self.tema):
            idx = np.where(self.tema == t)[0]
            orden = idx[np.argsort(-self.peso[idx] * rng.uniform(0.6, 1.4, len(idx)))]
            total_t, carga = self.peso[idx].sum(), np.zeros(3)
            for i in orden:
                s = int(np.argmax(self.p * total_t - carga))
                a[i] = s
                carga[s] += self.peso[i]
        return a

    def puntaje(self, a: np.ndarray) -> float:
        """Desvío total respecto de las proporciones objetivo, relativo al tamaño de cada categoría (menor es mejor)."""
        cuentas = np.eye(3)[a].T @ self.matriz
        return float((np.abs(cuentas - self.p[:, None] * self.totales[None, :])
                      / np.maximum(self.totales, 100)[None, :]).sum())


def particionar(df: pd.DataFrame, n_semillas=config.N_SEMILLAS, tope=config.TOPE_FAMILIA) -> dict:
    """Devuelve {'df': catálogo con columnas familia y split, 'puntaje', 'semilla', 'usables': {split: n}}."""
    df = df.copy()
    df["dataset_id"] = df["dataset_id"].astype(str)
    datasets = datasets_de(df)
    datasets["familia"] = familias_de(datasets)
    df["familia"] = df["dataset_id"].map(datasets["familia"])

    reparto = Reparto(df, datasets, tope)
    mejor, mejor_p, mejor_semilla = None, np.inf, None
    for semilla in range(n_semillas):
        a = reparto.asignar(semilla)
        p = reparto.puntaje(a)
        if p < mejor_p:
            mejor, mejor_p, mejor_semilla = a, p, semilla

    split_de_familia = dict(zip(reparto.familias, [config.SPLITS[i] for i in mejor], strict=True))
    df["split"] = df["familia"].map(split_de_familia)
    usables = df[df["estado"].isin(config.ESTADOS_USABLES)]["split"].value_counts().to_dict()
    return {"df": df, "puntaje": mejor_p, "semilla": mejor_semilla, "usables": usables}

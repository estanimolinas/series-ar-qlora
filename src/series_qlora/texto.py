"""Normalización de texto y regla de armado de la consulta de búsqueda."""
from __future__ import annotations

import re
import unicodedata

STOP = frozenset({"de", "del", "la", "el", "los", "las", "en", "y", "a", "por", "que", "cual",
                  "fue", "cuanto", "es", "un", "una", "para", "con", "se", "al", "lo"})

MESES = frozenset({"enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                   "septiembre", "setiembre", "octubre", "noviembre", "diciembre"})
ORDINALES = frozenset({"primer", "segundo", "tercer", "cuarto"})
MULETILLAS = frozenset({"variacion", "interanual", "respecto", "anterior", "ano", "mismo", "ultimo",
                        "ultima", "reciente", "dato", "cual", "fue", "cuanto"})


def norm(s) -> str:
    """Minúsculas, sin tildes, '_' como espacio y espacios colapsados. None/NaN -> ''."""
    if s is None or (isinstance(s, float) and s != s):
        s = ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.lower().replace("_", " ")).strip()


def consulta_de(pregunta) -> str:
    """Consulta de búsqueda derivada de la pregunta (regla v2, elegida sobre train).

    Conserva las palabras de contenido en su forma original y descarta: tokens con dígitos,
    meses, ordinales de trimestre, muletillas de la pregunta, stopwords, tokens de una letra
    y repetidos.
    """
    salida, vistos = [], set()
    for t in re.findall(r"\w+", str(pregunta)):
        n = norm(t)
        if (re.search(r"\d", t) or n in MESES or n in ORDINALES or n in MULETILLAS
                or n in STOP or len(n) <= 1 or n in vistos):
            continue
        vistos.add(n)
        salida.append(t)
    return " ".join(salida)

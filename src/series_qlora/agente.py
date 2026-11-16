"""Agente de herramientas sin framework: search_series -> get_series -> submit_answer.

El bucle no sabe nada del modelo ni de Langfuse: recibe `generar(mensajes)` y un trazador. Eso permite
probarlo con generadores guionados (incluido un oráculo) sin GPU.
"""
from __future__ import annotations

import json
import re

from .config import MAX_PASOS
from .observabilidad import SinTraza

# Prompt de sistema del baseline (modelo base). El modelo entrenado se evalúa SIN prompt de sistema.
PROMPT_OPT = """Sos un asistente que responde preguntas sobre estadísticas oficiales de Argentina usando tres herramientas. Seguí siempre este procedimiento:
1. Llamá a search_series con 2 a 6 palabras clave del tema (sin fechas ni números). Elegí, entre los candidatos devueltos, la serie cuyo título, distribución y frecuencia (mensual, trimestral o anual) mejor coincidan con lo que se pide. Usá solo ids devueltos por search_series; nunca inventes ids.
2. Llamá a get_series con la serie elegida:
- Valor de un período: start_date y end_date iguales, en formato AAAA (series anuales) o AAAA-MM.
- Último dato disponible: limit=1 y sort="desc", sin fechas.
- Variación interanual: representation_mode="percent_change_a_year_ago", start_date un año antes del período pedido y end_date el período pedido, con fechas completas AAAA-MM-DD (día 01; en series trimestrales, el primer mes del trimestre). La API devuelve una fracción (0.05 equivale a 5%) y la última fila corresponde al período pedido.
3. Llamá a submit_answer con la respuesta, el valor y la fecha del dato (formato AAAA-MM, o AAAA si la serie es anual). Para variaciones, el valor va en porcentaje (la fracción multiplicada por 100).
Si get_series devuelve data vacía, la API no tiene datos de esa serie para ese período: entregá submit_answer con valor y fecha en null y explicalo en la respuesta. No inventes valores."""

# Formato Hermes de Qwen3: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
RE_LLAMADA = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parsear_llamada(texto: str):
    """Extrae (nombre, argumentos). None si no hay llamada; ValueError si el JSON o su forma son inválidos."""
    m = RE_LLAMADA.search(texto)
    if not m:
        return None
    try:
        llamada = json.loads(m.group(1))
        nombre, args = llamada["name"], llamada["arguments"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError("llamada con formato inválido") from e
    if not isinstance(args, dict):
        raise ValueError("arguments no es un objeto")
    return nombre, args


def episodio(pregunta, sistema, generar, herramientas, trazador=None, etiqueta_modelo="", max_pasos=MAX_PASOS):
    """Corre un episodio. Devuelve (log, mensajes).

    `generar(mensajes)` -> (texto | None, tokens_entrada, tokens_salida, segundos). Texto None = contexto excedido.
    """
    trazador = trazador or SinTraza()
    msgs = ([{"role": "system", "content": sistema}] if sistema else []) + [{"role": "user", "content": pregunta}]
    log = {"pasos": 0, "n_search": 0, "n_get": 0, "formato_invalido": False, "sin_llamada": False,
           "contexto_excedido": False, "submit": None, "ids_ultima": None, "ultimo_texto": ""}
    for paso in range(1, max_pasos + 1):
        log["pasos"] = paso
        with trazador.observar("generation", f"generacion-{paso}", model=etiqueta_modelo, input=msgs[-1]) as gen:
            txt, n_in, n_out, seg = generar(msgs)
            gen.update(output=txt, metadata={"tokens_entrada": n_in, "tokens_salida": n_out,
                                             "segundos": round(seg, 2)})
        if txt is None:
            log["contexto_excedido"] = True
            break
        txt = txt.replace("<|im_end|>", "").strip()
        log["ultimo_texto"] = txt
        try:
            llamada = parsear_llamada(txt)
        except ValueError:
            log["formato_invalido"] = True
            break
        if llamada is None:
            log["sin_llamada"] = True
            break
        nombre, args = llamada
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"call_{paso}", "type": "function",
                                     "function": {"name": nombre, "arguments": args}}]})
        if nombre == "submit_answer":
            log["submit"] = args
            break
        with trazador.observar("span", f"herramienta:{nombre}", input=args) as ts:
            if nombre == "search_series":
                log["n_search"] += 1
                q = args.get("query")
                res = (herramientas.search_series(q) if isinstance(q, str)
                       else json.dumps({"status": 400, "error": "query debe ser texto"}))
            elif nombre == "get_series":
                log["n_get"] += 1
                log["ids_ultima"] = str(args.get("ids"))
                res = herramientas.get_series(args)
            else:
                res = json.dumps({"status": 400, "error": f"herramienta desconocida: {nombre}"}, ensure_ascii=False)
            ts.update(output=res[:1500])
        msgs.append({"role": "tool", "tool_call_id": f"call_{paso}", "name": nombre, "content": res})
    return log, msgs

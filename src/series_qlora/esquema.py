"""Definición de las tres herramientas del agente, en el formato que consume la plantilla de chat de Qwen3."""
from __future__ import annotations

HERRAMIENTAS = [
    {"type": "function", "function": {
        "name": "search_series",
        "description": "Busca series estadísticas en el catálogo por palabras clave. Devuelve hasta 10 candidatas.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Palabras clave del tema, sin fechas."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_series",
        "description": "Consulta datos de una serie de la API de Series de Tiempo. Fechas como 'AAAA', 'AAAA-MM' o 'AAAA-MM-DD'.",
        "parameters": {"type": "object", "properties": {
            "ids": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "representation_mode": {"type": "string",
                                    "enum": ["value", "change", "percent_change", "percent_change_a_year_ago"]},
            "limit": {"type": "integer"},
            "sort": {"type": "string", "enum": ["asc", "desc"]}},
            "required": ["ids"]}}},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Entrega la respuesta final. 'valor' y 'fecha' van en null si la API no devolvió datos.",
        "parameters": {"type": "object", "properties": {
            "respuesta": {"type": "string"},
            "valor": {"type": ["number", "null"]},
            "fecha": {"type": ["string", "null"]}},
            "required": ["respuesta"]}}},
]

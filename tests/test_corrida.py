import json

import pandas as pd

from conftest import llamada
from series_qlora.corrida import cargar_resultados, ejecutar, leer_progreso, nombre_corrida, resumen


def preguntas_sinteticas():
    data = json.dumps([["2020-01-01", 1.5]])
    filas = []
    for i, tipo in enumerate(["ultimo_dato", "ultimo_dato", "sin_datos"]):
        params = {"ids": "ser_emae", "limit": 1, "sort": "desc"} if tipo != "sin_datos" else {"ids": "ser_emae"}
        filas.append({"spec_id": f"x-{i}", "tipo": tipo, "frecuencia": "mensual", "params": json.dumps(params),
                      "data": data if tipo != "sin_datos" else "[]", "tecnica": f"pregunta {i}"})
    return pd.DataFrame(filas)


def generador_oraculo():
    """Responde bien las de tipo ultimo_dato y con null las sin_datos (según el texto de la pregunta)."""
    estado = {"paso": 0}

    def generar(msgs):
        pregunta = [m for m in msgs if m["role"] == "user"][0]["content"]
        n = sum(1 for m in msgs if m["role"] == "tool")
        sin = pregunta.endswith("2")
        if n == 0:
            t = llamada("get_series", ids="ser_emae", **({} if sin else {"limit": 1, "sort": "desc"}))
        else:
            t = llamada("submit_answer", respuesta="x", valor=None if sin else 1.5, fecha=None if sin else "2020-01")
        estado["paso"] += 1
        return t, 10, 5, 0.0

    return generar


def test_nombre_corrida():
    assert nombre_corrida("entrenado", "test", "tecnica", False) == "finetuned_test_tecnica_completa_basico"
    assert nombre_corrida("base", "val", "tecnica", True) == "baseline_val_tecnica_prueba_optimizado"


def test_ejecutar_con_oraculo_acierta_todo_y_resume(tmp_path, herramientas, api):
    api.respuestas["ser_emae"] = {"data": [["2020-01-01", 1.5]]}
    R = ejecutar(preguntas_sinteticas(), corrida="entrenado", split="val", estilo="tecnica",
                 generar=generador_oraculo(), herramientas=herramientas, archivo_local=tmp_path / "r.jsonl",
                 log=lambda *a: None)
    assert len(R) == 3 and R["correcta"].all()
    assert "correcta: 3/3" in resumen(R, "oraculo")


def test_retoma_no_repite_y_descarta_la_linea_cortada(tmp_path, herramientas):
    archivo = tmp_path / "r.jsonl"
    kw = dict(corrida="entrenado", split="val", estilo="tecnica", generar=generador_oraculo(),
              herramientas=herramientas, archivo_local=archivo, log=lambda *a: None)
    ejecutar(preguntas_sinteticas().head(2), **kw)
    with open(archivo, "a") as f:
        f.write('{"resultado": {"spec_id": "x-9"')          # corte de sesión a mitad de línea
    assert leer_progreso(archivo) == {"x-0", "x-1"}
    assert archivo.read_text().endswith("\n") and "x-9" not in archivo.read_text()
    R = ejecutar(preguntas_sinteticas(), **kw)
    assert sorted(R["spec_id"]) == ["x-0", "x-1", "x-2"] and R["spec_id"].is_unique


def test_guardado_periodico_y_final(tmp_path, herramientas):
    llamadas = []
    ejecutar(preguntas_sinteticas(), corrida="base", split="val", estilo="tecnica", generar=generador_oraculo(),
             herramientas=herramientas, archivo_local=tmp_path / "r.jsonl", guardar=lambda: llamadas.append(1),
             cada=2, log=lambda *a: None)
    assert len(llamadas) == 2          # una al llegar a 2 preguntas y una al final


def test_resultados_y_trazas_se_guardan_juntos(tmp_path, herramientas):
    ejecutar(preguntas_sinteticas().head(1), corrida="entrenado", split="val", estilo="tecnica",
             generar=generador_oraculo(), herramientas=herramientas, archivo_local=tmp_path / "r.jsonl",
             log=lambda *a: None)
    R, trazas = cargar_resultados(tmp_path / "r.jsonl")
    assert len(R) == len(trazas) == 1 and trazas[0]["pregunta"] == "pregunta 0" and trazas[0]["messages"]

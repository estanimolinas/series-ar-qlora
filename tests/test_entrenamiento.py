import json

import pytest

from series_qlora.entrenamiento import (
    LORA,
    argumentos_sft,
    cargar_trayectorias,
    familias_de_control,
    pasos_por_epoca,
    separar,
)


def trayectorias(tamanos):
    return [{"qid": f"{f}-{i}", "familia": f} for f, n in tamanos.items() for i in range(n)]


TAMANOS = {f"f{i}": n for i, n in enumerate([3, 5, 2, 8, 4, 6, 1, 7, 3, 5, 2, 4, 6, 3, 1, 5, 4, 2, 6, 3])}


def test_el_control_se_aparta_por_familia_completa_y_no_se_solapa_con_entrenamiento():
    T = trayectorias(TAMANOS)
    train, control = separar(T)
    assert {t["familia"] for t in train}.isdisjoint({t["familia"] for t in control})
    assert len(train) + len(control) == len(T) and control


def test_el_control_reune_al_menos_el_5_por_ciento_sin_pasarse_de_una_familia():
    T = trayectorias(TAMANOS)
    n_control = sum(1 for t in T if t["familia"] in familias_de_control([t["familia"] for t in T]))
    assert 0.05 * len(T) <= n_control < 0.05 * len(T) + max(TAMANOS.values())


def test_el_control_nunca_incluye_una_familia_que_pese_mas_del_8_por_ciento():
    fam = [t["familia"] for t in trayectorias({"enorme": 50, **{f"p{i}": 2 for i in range(25)}})]
    assert all("enorme" not in familias_de_control(fam, semilla=s, tope_familia=0.08) for s in range(100))


def test_la_separacion_es_reproducible_y_depende_de_la_semilla():
    fam = [t["familia"] for t in trayectorias(TAMANOS)]
    assert familias_de_control(fam, 3407) == familias_de_control(fam, 3407)
    assert any(familias_de_control(fam, s) != familias_de_control(fam, 3407) for s in range(5))


def test_pasos_por_epoca_con_los_datos_del_entrenamiento_publicado():
    assert pasos_por_epoca(1586) == 100         # 1.586 ejemplos de entrenamiento, lote efectivo de 16


@pytest.mark.parametrize("bf16", [False, True])
def test_argumentos_sft(bf16, tmp_path):
    a = argumentos_sft(tmp_path, bf16=bf16)
    assert a["fp16"] is (not bf16) and a["bf16"] is bf16
    assert a["per_device_train_batch_size"] * a["gradient_accumulation_steps"] == 16
    assert a["max_steps"] == -1 and a["max_length"] == 2048 and a["learning_rate"] == 2e-4
    assert argumentos_sft(tmp_path, pasos_maximos=10)["max_steps"] == 10


def test_cargar_trayectorias_ignora_lineas_vacias(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"qid": "a"}) + "\n\n" + json.dumps({"qid": "b"}) + "\n", encoding="utf-8")
    assert [t["qid"] for t in cargar_trayectorias(p)] == ["a", "b"]


def test_parametros_del_adaptador_coinciden_con_los_del_registro_de_entrenamiento():
    # Dimensiones de Qwen3-4B: 36 capas, dimensión 2560, capa intermedia 9728, 32 cabezas de consulta y 8 de clave/valor de 128
    oculta, intermedia, capas, cabeza = 2560, 9728, 36, 128
    q, kv = 32 * cabeza, 8 * cabeza
    dimensiones = {"q_proj": (oculta, q), "k_proj": (oculta, kv), "v_proj": (oculta, kv), "o_proj": (q, oculta),
                   "gate_proj": (oculta, intermedia), "up_proj": (oculta, intermedia), "down_proj": (intermedia, oculta)}
    assert set(dimensiones) == set(LORA["target_modules"])
    por_capa = sum(LORA["r"] * (entrada + salida) for entrada, salida in dimensiones.values())
    assert por_capa == 917_504
    assert capas * por_capa == 33_030_144      # "Trainable parameters = 33,030,144 of 4,055,498,240 (0.81% trained)"

"""Entrenamiento QLoRA con Unsloth.

La preparación de los datos (separar el control por familias) y la configuración no necesitan GPU y se prueban;
`entrenar` carga el modelo en 4 bits y entrena (requiere GPU y las dependencias de requirements-colab.txt).
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

from .config import BASE_MODEL_ID, MAX_SEQ_ENTRENAMIENTO

SEMILLA = 3407
LORA = {"r": 16, "lora_alpha": 16, "lora_dropout": 0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
LOTE, ACUMULACION = 2, 8          # lote efectivo de 16 ejemplos por paso
PARTE_USUARIO, PARTE_ASISTENTE = "<|im_start|>user\n", "<|im_start|>assistant\n"


def cargar_trayectorias(ruta) -> list[dict]:
    return [json.loads(linea) for linea in Path(ruta).read_text(encoding="utf-8").splitlines() if linea.strip()]


def familias_de_control(familias: list, semilla=SEMILLA, fraccion=0.05, tope_familia=0.08) -> set:
    """Familias que se apartan para medir la pérdida en datos que no se entrenan.

    Se barajan las familias con una semilla fija y se van apartando hasta reunir ~`fraccion` de las trayectorias,
    salteando las que por sí solas pesan más de `tope_familia` (apartar una familia enorme dejaría a train sin ella).
    Se aparta por familia, no por ejemplo, para que el control no contenga series casi idénticas a las entrenadas.
    """
    cuenta = Counter(familias)
    orden = sorted(cuenta)
    random.Random(semilla).shuffle(orden)
    control, apartadas = set(), 0
    for f in orden:
        if apartadas >= fraccion * len(familias):
            break
        if cuenta[f] > tope_familia * len(familias):
            continue
        control.add(f)
        apartadas += cuenta[f]
    return control


def separar(trayectorias: list[dict], semilla=SEMILLA) -> tuple[list[dict], list[dict]]:
    """(entrenamiento, control) con el control separado por familia."""
    control = familias_de_control([t["familia"] for t in trayectorias], semilla)
    return ([t for t in trayectorias if t["familia"] not in control],
            [t for t in trayectorias if t["familia"] in control])


def pasos_por_epoca(n_ejemplos: int) -> int:
    return math.ceil(n_ejemplos / (LOTE * ACUMULACION))


def argumentos_sft(salida, epocas=1, pasos_maximos=-1, bf16=False, semilla=SEMILLA,
                   max_seq=MAX_SEQ_ENTRENAMIENTO) -> dict:
    """Hiperparámetros del entrenamiento (argumentos de `trl.SFTConfig`).

    lr 2e-4 con decaimiento lineal y 10 pasos de calentamiento: valores habituales para LoRA/QLoRA; no se probaron
    alternativas. AdamW de 8 bits para ahorrar memoria. Se guarda y se mide el control cada 50 pasos.
    """
    return dict(
        per_device_train_batch_size=LOTE, gradient_accumulation_steps=ACUMULACION, num_train_epochs=epocas,
        max_steps=pasos_maximos, learning_rate=2e-4, lr_scheduler_type="linear", warmup_steps=10,
        weight_decay=0.01, optim="adamw_8bit", fp16=not bf16, bf16=bf16, logging_steps=10, seed=semilla,
        output_dir=str(salida), report_to="none", dataset_text_field="text", max_length=max_seq,
        save_strategy="steps", save_steps=50, save_total_limit=2, eval_strategy="steps", eval_steps=50,
        per_device_eval_batch_size=LOTE)


def entrenar(trayectorias: list[dict], herramientas: list, salida, modo="completo", epocas=1, semilla=SEMILLA):
    """Entrena el adaptador LoRA y lo guarda en `salida/adapter` (modo 'completo').

    modo: 'prueba_10_pasos' (solo para comprobar que todo corre) o 'completo'. Si hay checkpoints en `salida`,
    el modo completo retoma desde el último. Devuelve el `trainer` entrenado.
    """
    os.environ["UNSLOTH_DISABLE_CPU_FALLBACK"] = "1"      # que falle en vez de entrenar en CPU
    # unsloth debe importarse antes que torch/transformers/trl
    from unsloth import FastLanguageModel  # noqa: I001
    from unsloth.chat_templates import train_on_responses_only  # noqa: I001
    import bitsandbytes as bnb  # noqa: I001
    import torch  # noqa: I001
    from datasets import Dataset  # noqa: I001
    from trl import SFTConfig, SFTTrainer  # noqa: I001

    if not torch.cuda.is_available():
        raise RuntimeError("No hay GPU: Runtime > Change runtime type > T4 GPU")
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)

    modelo, tokenizador = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID, max_seq_length=MAX_SEQ_ENTRENAMIENTO, load_in_4bit=True)
    if modelo.max_seq_length != MAX_SEQ_ENTRENAMIENTO:
        raise RuntimeError(f"El modelo quedó con max_seq_length={modelo.max_seq_length}")
    modelo = FastLanguageModel.get_peft_model(modelo, **LORA, use_gradient_checkpointing="unsloth", random_state=semilla)
    # Sin cuantización a 4 bits no sería QLoRA: se comprueba y se muestra qué se entrena
    if sum(isinstance(m, bnb.nn.Linear4bit) for m in modelo.modules()) == 0:
        raise RuntimeError("El modelo base no está cuantizado a 4 bits")
    entrenables = [n for n, p in modelo.named_parameters() if p.requires_grad]
    if not all("lora_" in n for n in entrenables):
        raise RuntimeError("Hay parámetros entrenables que no son del adaptador LoRA")

    train, control = separar(trayectorias, semilla)

    def texto(t):
        return tokenizador.apply_chat_template(t["messages"], tools=herramientas, tokenize=False)

    ds_train = Dataset.from_list([{"text": texto(t)} for t in train])
    ds_control = Dataset.from_list([{"text": texto(t)} for t in control])
    largo_max = max(len(tokenizador(x["text"], add_special_tokens=False)["input_ids"]) for x in ds_train)
    if largo_max > MAX_SEQ_ENTRENAMIENTO:
        raise ValueError(f"Hay ejemplos de {largo_max} tokens (límite {MAX_SEQ_ENTRENAMIENTO})")
    print(f"Entrenamiento: {len(ds_train)} | control: {len(ds_control)} | tokens máx: {largo_max}")

    config = SFTConfig(**argumentos_sft(salida, epocas, 10 if modo == "prueba_10_pasos" else -1,
                                        torch.cuda.is_bf16_supported(), semilla))
    trainer = SFTTrainer(model=modelo, processing_class=tokenizador, train_dataset=ds_train,
                         eval_dataset=ds_control, args=config)
    # La pérdida se calcula solo sobre lo que escribe el asistente, no sobre las respuestas de las herramientas
    trainer = train_on_responses_only(trainer, instruction_part=PARTE_USUARIO, response_part=PARTE_ASISTENTE)

    t0 = time.time()
    retomar = modo == "completo" and bool(glob.glob(str(salida / "checkpoint-*")))
    resultado = trainer.train(resume_from_checkpoint=retomar)
    print(f"Tiempo: {(time.time() - t0) / 60:.1f} min | pérdida de entrenamiento: {resultado.metrics.get('train_loss')}")
    log = trainer.state.log_history
    print("Pérdida de control (paso, valor):", [(h["step"], round(h["eval_loss"], 4)) for h in log if "eval_loss" in h])
    print("Pérdida de entrenamiento (paso, valor):", [(h["step"], round(h["loss"], 4)) for h in log if "loss" in h])
    if modo == "completo":
        modelo.save_pretrained(str(salida / "adapter"))
        tokenizador.save_pretrained(str(salida / "adapter"))
    return trainer

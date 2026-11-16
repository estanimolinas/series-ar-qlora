"""Carga del modelo y generación (requiere GPU y unsloth; no se importa en los tests)."""
from __future__ import annotations

import os
import time

from .config import BASE_MODEL_ID, MAX_NUEVOS, MAX_SEQ_INFERENCIA


def cargar(corrida: str, ruta_adaptador: str, max_seq: int = MAX_SEQ_INFERENCIA):
    """Carga el modelo en 4 bits. corrida: 'entrenado' (adaptador QLoRA) o 'base' (sin adaptador)."""
    if corrida not in ("entrenado", "base"):
        raise ValueError("corrida debe ser 'entrenado' o 'base'")
    os.environ["UNSLOTH_DISABLE_CPU_FALLBACK"] = "1"
    # unsloth debe importarse antes que torch/transformers
    from unsloth import FastLanguageModel  # noqa: I001
    import torch  # noqa: I001

    if not torch.cuda.is_available():
        raise RuntimeError("No hay GPU: Runtime > Change runtime type > T4 GPU")
    nombre = ruta_adaptador if corrida == "entrenado" else BASE_MODEL_ID
    if corrida == "entrenado" and not os.path.exists(os.path.join(ruta_adaptador, "adapter_config.json")):
        raise FileNotFoundError(f"No encuentro el adaptador en {ruta_adaptador}")
    modelo, tokenizador = FastLanguageModel.from_pretrained(
        model_name=nombre, max_seq_length=max_seq, load_in_4bit=True)
    FastLanguageModel.for_inference(modelo)
    # Unsloth puede ignorar el max_seq_length pedido: se verifica el valor real
    if modelo.max_seq_length != max_seq:
        raise RuntimeError(f"El modelo quedó con max_seq_length={modelo.max_seq_length} (se pidió {max_seq})")
    return modelo, tokenizador


def crear_generador(modelo, tokenizador, herramientas: list, max_nuevos: int = MAX_NUEVOS):
    """Devuelve generar(mensajes) -> (texto | None, tokens_entrada, tokens_salida, segundos).

    Decodificación determinista. Si el prompt deja menos de `max_nuevos` tokens de contexto, devuelve texto None.
    """
    import torch

    def generar(mensajes):
        ids = tokenizador.apply_chat_template(
            mensajes, tools=herramientas, add_generation_prompt=True, return_tensors="pt").to("cuda")
        n_in = ids.shape[1]
        if n_in > modelo.max_seq_length - max_nuevos:
            return None, n_in, 0, 0.0
        t = time.time()
        with torch.no_grad():
            out = modelo.generate(ids, max_new_tokens=max_nuevos, do_sample=False)
        nuevos = out[0][n_in:]
        return tokenizador.decode(nuevos, skip_special_tokens=False), n_in, len(nuevos), time.time() - t

    return generar

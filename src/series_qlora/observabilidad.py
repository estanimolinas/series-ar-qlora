"""Trazado de episodios. El agente solo conoce esta interfaz; Langfuse es una implementación opcional."""
from __future__ import annotations


class _ObservacionNula:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, **kwargs):
        return None


class SinTraza:
    """Trazador que no hace nada (tests y corridas sin Langfuse)."""

    def observar(self, as_type, name, **kwargs):
        return _ObservacionNula()

    def id_traza_actual(self):
        return None

    def puntuar(self, observacion, nombre, valor):
        """Devuelve None si salió bien o un texto con el error."""
        return None

    def flush(self):
        return None


class TrazadorLangfuse:
    """Envuelve un cliente de Langfuse (SDK >= 3: start_as_current_observation)."""

    def __init__(self, cliente):
        self.lf = cliente

    def observar(self, as_type, name, **kwargs):
        return self.lf.start_as_current_observation(as_type=as_type, name=name, **kwargs)

    def id_traza_actual(self):
        return self.lf.get_current_trace_id()

    def puntuar(self, observacion, nombre, valor):
        try:
            observacion.score_trace(name=nombre, value=float(valor))
            return None
        except Exception as e:  # el score no debe romper la corrida
            return f"{type(e).__name__}: {str(e)[:100]}"

    def flush(self):
        self.lf.flush()

# Uso de la API de Anthropic

La API se usó **solo para construir datos y verificar calidad**, nunca como agente del sistema evaluado.

| Uso | Modelo | Razonamiento | Salida |
|---|---|---|---|
| Preguntas en estilo técnico y coloquial a partir de la ficha de cada serie | Claude Haiku 5.5 (`claude-haiku-5-5`) | activado (`max_tokens` 3000) | Una pregunta por (serie, tipo, estilo) |
| Verificador de fidelidad: ¿la pregunta describe exactamente el dato? | idem | desactivado | Marca de fidelidad por pregunta; las marcadas se excluyen del entrenamiento |
| Consultas de búsqueda (variante v1, luego reemplazada por la regla `texto.consulta_de`) | idem | desactivado | Texto de consulta |

Pasos reproducibles con `scripts/preparar_preguntas.py` (`preguntas`, `verificar`). Costos medidos al construir los conjuntos, en corridas retomadas (por eso no hay un total por conjunto):

| Corrida | Costo (USD) |
|---|---|
| Preguntas de train, 158 especificaciones pendientes | 0,089 |
| Preguntas de test, 432 especificaciones | 0,248 |
| Verificación de fidelidad: train (1.199) / val (312) / test (608) | 0,101 / 0,029 / 0,052 |

Precio de referencia usado para presupuestar: 0,10 USD (entrada) y 0,50 USD (salida) por millón de tokens.
Presupuesto fijado: 5 USD. Gasto total del proyecto: 1,75 USD (valor informado por el autor; la fuente de verdad es la consola de Anthropic).

Controles:
- Las claves se leen de variables de entorno / Secrets de Colab; no hay claves en el repositorio.
- Un fallo `401` a mitad de una corrida (clave inválida) dejó 26 de 94 pedidos sin generar; la celda lo detectó y se detuvo en lugar de calcular métricas con datos incompletos.

Términos de uso (Términos Comerciales de Anthropic, vigentes desde el 17 de junio de 2025, https://www.anthropic.com/legal/commercial-terms): la sección D.4 prohíbe acceder a los Servicios "to build a competing product or service, including to train competing AI models ... except as expressly approved by Anthropic"; la sección B cede al cliente los derechos sobre las salidas, sujeto al cumplimiento de los términos. El modelo ajustado aquí es de terceros y de uso
de investigación, pero **la lectura de esos términos debe confirmarse antes de publicar el adaptador** (ver `datos_y_licencias.md`).

# Decisiones de diseño

| Decisión | Alternativa descartada | Motivo |
|---|---|---|
| QLoRA (base en 4 bits + adaptadores LoRA) | LoRA en 16 bits | El objetivo del proyecto es QLoRA; cabe en una T4 de 15 GB (3,7 GB de memoria con 33 M de parámetros entrenables, 0,81 %) |
| LoRA r=16, alpha=16, dropout 0, sobre q/k/v/o/gate/up/down | Solo atención | Cubre todas las proyecciones lineales con costo bajo |
| 1 época (100 pasos, lote efectivo 16, lr 2e-4 lineal, `adamw_8bit`, fp16) | Más épocas | T4 sin bf16; el control (familias aparte) no mostró señal de sobreajuste en el punto de control 50 (pérdida de entrenamiento 0,127 vs. validación 0,129) |
| Pérdida solo sobre las respuestas del asistente (`train_on_responses_only`) | Pérdida sobre todo el texto | No hay que aprender a reproducir las respuestas de las herramientas (verificado: 0 ejemplos entrenan un `<tool_response>`) |
| Agente propio sin framework | LangChain / ADK | Control total sobre el formato de llamadas y sobre qué se mide |
| Buscador SQLite FTS5 + bm25 | Embeddings | Sin dependencias, determinista, rápido; embeddings quedan como mejora futura |
| Caché de la API en SQLite | Llamar a la API en cada corrida | Reproducibilidad y velocidad; incluye los 400 |
| Partición por **familias** de datasets (semilla 3479, similitud ≥ 0,85 más uniones manuales) | Partición aleatoria por series | Series de un mismo dataset son casi duplicadas: una partición aleatoria filtraría información a test |
| Evaluación con agente oráculo y controles negativos antes de medir modelos | Confiar en el evaluador | Un evaluador sin validar vuelve inservible cualquier comparación |
| Test se evalúa una sola vez, sin ajustes posteriores | Iterar mirando test | Evita sobreajustar a test |
| Las preguntas las redacta un modelo, pero las acepta el código (`validar`: período, números, largo, no copiar el título, "interanual") y las marcas de calidad también salen de código | Aceptar lo que devuelve el modelo | El modelo omite o inventa períodos; con la regla, los rechazos quedan contados y los reintentos son baratos |
| Un segundo paso de verificación de fidelidad (otra llamada, sin razonamiento) que excluye las técnicas que cambian el sentido del dato y marca las coloquiales ambiguas | Revisión manual de todas | Escala a miles de preguntas; la revisión humana exhaustiva no se hizo (ver limitaciones) |
| Las especificaciones salen de la respuesta real de la API y la variación interanual se verifica contra el cálculo propio con los valores crudos | Calcularla solo con la representación de la API | Descarta casos en que la API y los valores crudos no coinciden, o con valores no positivos |

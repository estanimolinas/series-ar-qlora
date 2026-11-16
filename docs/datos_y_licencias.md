# Datos y licencias

Documento de control antes de publicar. Nada de lo marcado como _pendiente_ está decidido.

| Elemento | Origen | Se publica | Estado |
|---|---|---|---|
| Catálogo y series | API de Series de Tiempo, datos.gob.ar | Solo identificadores y resultados agregados; los datos crudos no van al repositorio | El repositorio de la API (`datosgobar/series-tiempo-ar-api`) es software con licencia MIT; su README no dice qué licencia tienen los datos que sirve. _Pendiente:_ confirmar la licencia de los datos en el portal (datos.gob.ar) y citarla |
| Preguntas generadas | Claude Haiku 5.5 | _Pendiente_ | Revisar restricciones de los Términos Comerciales de Anthropic (ver `uso_api_anthropic.md`) |
| Modelo base | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` | No (se referencia) | Verificar la licencia del modelo base |
| Adaptador QLoRA | Entrenado en este proyecto | _Pendiente_ (Hugging Face, solo el adaptador) | Depende de las dos filas anteriores |
| Código | Este repositorio | Sí | Licencia MIT (`LICENSE`) |

Propósito del proyecto: personal, educativo y de portfolio; sin fines comerciales (no se vende ni se ofrece como producto o servicio) y sin intención de competir con ningún modelo o servicio de terceros. Esto aclara la intención, pero no reemplaza la lectura de los términos de cada tercero: en particular, la sección D.4 de los Términos Comerciales de Anthropic (ver `uso_api_anthropic.md`) sigue siendo una cuestión abierta para publicar el adaptador o las preguntas.

Regla del proyecto: no se usan datos ni dominio de ningún empleador; todo el material proviene de datos abiertos públicos.

# Errores corregidos durante el proyecto

Registro de lo que salió mal y cómo se resolvió. Sirve para auditar el resultado final.

## Infraestructura y entrenamiento

| Síntoma | Causa | Corrección |
|---|---|---|
| `401` de Langfuse | Host de la región UE con una cuenta de EE.UU.; además `get_client()` queda fijo a la primera configuración | Usar `https://us.cloud.langfuse.com` y reiniciar la sesión |
| "no GPU found, training on CPU" | Runtime sin GPU | `UNSLOTH_DISABLE_CPU_FALLBACK=1` y comprobación explícita de `torch.cuda.is_available()` |
| `RuntimeError` por entrada > 2048 tokens | El modelo se cargó con `max_seq_length=2048` aunque se pidió 4096; asignar el atributo no alcanza | Recargar y verificar `model.max_seq_length` real (`modelo.cargar` lo comprueba) |
| `SFTConfig` rechazó `warmup_ratio` | Cambio de API | `warmup_steps=10`; el script descarta argumentos no aceptados e informa cuáles |
| Checkpoints pisados | La prueba de 10 pasos escribió en la misma carpeta | Carpeta de salida por modo (`qlora_<modo>`) |
| Resultados del base y del ajustado podían pisarse | Mismo nombre de archivo | Prefijos `baseline_` / `finetuned_` y prompt en el nombre (`corrida.nombre_corrida`) |

## Evaluación

| Hallazgo | Corrección |
|---|---|
| Intervalo de Wilson con límite inferior de −2,8e-17 cuando k = 0 (error de redondeo de punto flotante; el valor exacto es 0) | Recorte a [0, 1] (detectado por los tests al migrar a módulos; ver `guia_metricas.md`, sección 3) |
| Variación interanual: la API devuelve fracción y el gold está en porcentaje | El evaluador estricto exige porcentaje; la versión `flex` acepta la fracción y se reporta aparte |
| Cortes de sesión dejaban líneas truncadas en los resultados | Guardado cada 25 preguntas y retoma que descarta la última línea incompleta (`corrida.leer_progreso`, con test) |
| Al migrar a módulos, el índice de búsqueda se reconstruyó con columnas distintas de las usadas en las corridas (título y distribución, en lugar de título y un contexto con distribución, descripción, dataset, tema, unidades y frecuencia) | `herramientas.textos_indice` y `construir_indice` reproducen el índice original; se comprobó contra las columnas `txt_titulo` y `txt_contexto` del catálogo guardado (8.226 de 8.226 series iguales) |
| Al reconstruir las trayectorias con el catálogo anterior al dataset de tipo de cambio (8.209 series), 6 de 1.705 diferían de las usadas para entrenar | Las trayectorias originales se armaron con el catálogo completo (8.226): con 17 documentos más cambia el IDF de bm25 y se reordenan empates en los últimos puestos del top 10 (en 5 casos entra la serie `168.1_T_CAMBI500_D_0_0_17`). Con el catálogo completo coinciden las 1.705 completas; la serie objetivo y su rango no cambiaban en ninguno de los 6 |
| El orden del top 10 dependía de la máquina: tres series de telefonía tienen el mismo puntaje bm25 matemático, pero SQLite 3.53 en arm64 (macOS) lo calcula con un bit de diferencia en el último dígito (`-31.001623177371282` contra `-31.00162317737128`) y en x86 sale idéntico. Con `ORDER BY s` el orden entre ellas cambiaba y 6 de 1.705 trayectorias salían con el top 3 en otro orden | `ORDER BY round(s, 9), rowid`: se redondea el puntaje y se desempata por fila del catálogo. Una prueba con el mismo desfase de un bit reproduce el orden erróneo sin redondeo y el correcto con redondeo; la verificación completa en macOS la hace `--comparar`. Los resultados publicados se obtuvieron con el orden de SQLite en Colab (x86): en las 1.705 trayectorias coincide con el de la regla nueva. En test se repitieron con la regla nueva las 1.170 búsquedas registradas (585 por modelo) y las 1.170 devuelven el mismo top 10 en el mismo orden; en validación no se pudo verificar porque no hay episodios de validación en el repositorio local |

## Hipótesis propias que resultaron falsas

- "El modelo aprendió a elegir el primer candidato del buscador": no se sostuvo al analizar las selecciones.
- "Los fallos se explican por el vocabulario de Haiku": no se sostuvo.
- Una lista de palabras de "ruido" para la consulta incluía términos que en realidad son señal (mensual, millones).
- La caída en la calidad de selección observada en validación no se replicó en test.
- Estimaciones de tiempo: corrida de validación (se estimó 15–30 min, tomó ~50) y entrenamiento (se estimó 20–40 min, tomó ~75).

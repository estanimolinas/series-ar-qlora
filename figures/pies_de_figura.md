**Figura 1.** Porcentaje de respuestas correctas en test (n = 585), en total y por tipo de pregunta. Barras violeta: modelo base con prompt optimizado; barras verdes: modelo ajustado con QLoRA, sin prompt de sistema. Barras de error: intervalo de confianza del 95 % de Wilson, que supone preguntas independientes. Diferencia total entre modelos: +44,4 puntos; IC del 95 % remuestreando las 23 familias de datasets (10.000 réplicas): +37,2 a +51,9.

**Figura 2.** Causa de fallo de las preguntas incorrectas en test (n = 585). Cada pregunta incorrecta se asigna a su primera causa, en este orden: no terminó con `submit_answer`, serie equivocada, valor equivocado y fecha equivocada. Dentro de valor equivocado se cuenta aparte el caso en que el valor es correcto expresado como fracción en vez de porcentaje.

**Figura 3.** Porcentaje de respuestas correctas en validación (n = 299) y en test (n = 585). Barras de error: IC del 95 % de Wilson. Los niveles no son comparables entre conjuntos, porque contienen familias de datasets distintas.

**Figura 4.** Pérdida durante el entrenamiento QLoRA (1 época, 100 pasos), copiada del registro de la corrida. La pérdida de entrenamiento se registra cada 10 pasos; la de control, solo en los pasos 50 y 100, sobre 3 familias de datasets apartadas.

**Figura 5.** Tabla de contingencia emparejada en test (n = 585): cada pregunta según si la respondió correctamente el modelo base (filas) y el modelo ajustado (columnas). Ambas correctas: 142; solo el base: 24; solo el ajustado: 284; ninguna: 135.

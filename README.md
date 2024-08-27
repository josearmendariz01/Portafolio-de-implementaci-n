# Portafolio de Implementación

Este proyecto implementa un modelo de regresión logística para predecir si una persona presentará problemas cardíacos en los próximos 10 años.

## Tabla de Contenidos

- [Funcionamiento del Código: Funciones y Clases](#funcionamiento-del-código-funciones-y-clases)
- [Funcionamiento del Código: Regresión Logística](#funcionamiento-del-código-regresión-logística)
- [Funcionamiento del Código: Regresión Logística Multiclase](#funcionamiento-del-código-regresión-logística-multiclase)
- [Conclusiones](#conclusiones)

## Funcionamiento del Código: Funciones y Clases

El código está documentado con comentarios detallados, pero a grandes rasgos:

- **`dividir_datos_y_target`**: Función para dividir y mezclar aleatoriamente el conjunto de datos en entrenamiento y prueba, separando las características (features) de las etiquetas (labels).
- **Clase `EscaladorDatos`**: Escala y transforma los datos de las características para mejorar el rendimiento del modelo y acelerar la convergencia.
- **`sigmoid_function`**: Implementación de la función sigmoide, utilizada para convertir la salida lineal en una probabilidad.
- **`log_regression`**: Función para encontrar los pesos óptimos para la función sigmoide, utilizados para hacer predicciones y calcular probabilidades en regresiones logísticas, incluyendo la multiclase.

## Funcionamiento del Código: Regresión Logística

1. **Carga de datos**: Los valores `NaN` de cada característica se reemplazan con la media correspondiente.
2. **Selección de características**: Todas las características del conjunto de datos se utilizan excepto `TenYearCHD`, que es la variable objetivo (target).
3. **Creación de variables objetivo**: Se crean dos variables objetivo que indican riesgo de fallo cardíaco en 10 años:
   - 1 si hay riesgo, 0 si no.
   - Variable inversa para el segundo modelo.
4. **División del conjunto de datos**: Se divide el conjunto de datos en entrenamiento y prueba.
5. **Escalado y transformación de características**: Los datos se escalan y transforman.
6. **Optimización de parámetros**: La función `log_regression` encuentra los parámetros óptimos (theta).
7. **Predicción**: Se aplica la función sigmoide y se establece un umbral para clasificar las predicciones:
   - Valores ≥ 0.5 se redondean a 1.
   - Valores < 0.5 se redondean a 0.
8. **Cálculo de precisión**:
   - **Entrenamiento**: 85.79% de precisión para predecir que no habrá problemas cardíacos.
   - **Prueba**: 84.18% de precisión para predecir que no habrá problemas cardíacos.
   - **Entrenamiento**: 85.34% de precisión para predecir que sí habrá problemas cardíacos.
   - **Prueba**: 83.94% de precisión para predecir que sí habrá problemas cardíacos.

## Funcionamiento del Código: Regresión Logística Multiclase

1. **Diccionario de datos**: Se crea un diccionario para almacenar datos de personas sin problemas cardíacos en 10 años y dos subdiccionarios para probabilidades predichas y valores reales.
2. **Ciclo de predicciones**: Para cada elemento del diccionario:
   - **División de datos**: Entrenamiento y prueba.
   - **Escalado y transformación**: Mejorando rendimiento.
   - **Optimización**: Con `log_regression`.
   - **Predicción**: Con la función sigmoide.
   - **Almacenamiento de probabilidades**: En el diccionario correspondiente.
   - **Impresión de resultados**: Se imprime la probabilidad del evento y se verifica la precisión.

## Conclusiones

- **Precisión**: Los resultados muestran que el modelo de regresión logística es adecuado para este problema. La similitud en las precisiones de los conjuntos de entrenamiento y prueba indica un buen equilibrio entre sesgo y varianza.
- **Verificación**: Se verificó la precisión utilizando la función `accuracy_score` de **scikit-learn**. Después de confirmar los resultados, la librería fue eliminada para demostrar que no se utilizó ningún framework en este proyecto.
- **Regresión logística multiclase**: Se implementó para mostrar que es posible presentar los resultados en forma de probabilidades directas, permitiendo decisiones más informadas.

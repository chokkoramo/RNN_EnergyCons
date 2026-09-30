# RNN Energy Consumption App

Aplicación interactiva en **Streamlit** para explicar y probar una red neuronal recurrente (RNN) que estima el consumo de energía diario con patrón semanal.

## Contenido

La app incluye 5 secciones:

1. **Datos y ventanas**: visualización del consumo y formación de ventanas temporales.
2. **Una neurona**: simulación manual de una neurona recurrente.
3. **Estructura de la red**: diagrama de la arquitectura RNN + Dense.
4. **Evaluación**: comparación del modelo contra una línea base.
5. **Estimar el día siguiente**: predicción a partir de una ventana de entrada.

## Requisitos

- Python 3.10+ (recomendado)
- Dependencias de `requirements.txt`

## Instalación

```bash
pip install -r requirements.txt
```

## Archivos de modelo requeridos

La app necesita ambos archivos:

- `modelo_rnn.keras`
- `escala.json`

Puedes cargarlos desde la barra lateral o dejarlos en la raíz del proyecto para carga automática.

`escala.json` debe incluir al menos estas claves:

- `p_min`
- `p_max`
- `ventana`

## Ejecución

Desde la raíz del repositorio:

```bash
streamlit run app.py
```

## Notas

- Si faltan los archivos del modelo, la app pedirá subirlos.
- Las predicciones son más confiables dentro del rango de entrenamiento indicado por `p_min` y `p_max`.

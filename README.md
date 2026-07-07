# ft_linear_regression

Introducción práctica al machine learning mediante una regresión lineal simple.
El programa aprende la relación entre el kilometraje de un coche y su precio
utilizando descenso de gradiente.

## Objetivo

El proyecto se divide en dos programas:

1. `train.py` lee `data.csv`, calcula `theta0` y `theta1` y guarda el modelo.
2. `predict.py` recibe un kilometraje y estima el precio con el modelo guardado.

La fórmula de predicción es:

```text
estimatePrice(mileage) = theta0 + theta1 * mileage
```

Antes de entrenar, ambos parámetros valen `0`, por lo que la estimación también
es `0`.

## Estructura

```text
Ft_linear_regresion/
├── data.csv
├── model.json
├── Makefile
├── README.md
├── plots/                  # Se genera al ejecutar el bonus
├── src/
│   ├── linear_regression.py
│   ├── predict.py
│   ├── train.py
│   └── visualization.py
└── tests/
    └── test_linear_regression.py
```

- `linear_regression.py`: lectura de datos, fórmula, entrenamiento y modelo.
- `train.py`: interfaz del programa de entrenamiento.
- `predict.py`: interfaz del programa de estimación.
- `model.json`: parámetros compartidos por los dos programas.
- `tests/`: comprobaciones básicas del cálculo.

## Requisitos

- Python 3.9 o posterior.
- No se utilizan librerías externas ni librerías de machine learning.

## Uso

Desde la raíz del proyecto:

```bash
python3 src/predict.py 100000
```

Como el modelo empieza con parámetros a cero, antes de entrenar mostrará:

```text
Estimated price: 0.00
```

Entrena el modelo:

```bash
python3 src/train.py
```

Después, vuelve a solicitar una estimación:

```bash
python3 src/predict.py 100000
```

También se puede ejecutar el predictor de forma interactiva:

```bash
python3 src/predict.py
```

Atajos disponibles:

```bash
make train
make bonus
make predict
make test
make reset
```

`make reset` devuelve los parámetros del modelo a cero.

## Bonus

El bonus añade una evaluación de la precisión y dos visualizaciones sin
necesitar Matplotlib, NumPy ni ninguna dependencia externa:

```bash
make bonus
```

También se puede ejecutar directamente:

```bash
python3 src/train.py --bonus
```

El terminal muestra estas métricas:

- **MSE**: error cuadrático medio.
- **RMSE**: raíz del error cuadrático medio, expresada en unidades de precio.
- **MAE**: error absoluto medio.
- **R²**: proporción de variación explicada por el modelo; cuanto más cerca de
  `1`, mejor es el ajuste.

Además, se generan dos archivos que pueden abrirse en cualquier navegador:

- `plots/regression.svg`: puntos del dataset y recta aprendida.
- `plots/loss.svg`: evolución del error durante el descenso de gradiente.

## Opciones de entrenamiento

El número de iteraciones y el learning rate se pueden modificar:

```bash
python3 src/train.py --iterations 2000 --learning-rate 0.05
```

Para consultar todas las opciones:

```bash
python3 src/train.py --help
python3 src/predict.py --help
```

## Cómo funciona el entrenamiento

Para cada iteración:

1. Se calcula el precio estimado de todos los coches.
2. Se obtiene el error entre cada estimación y su precio real.
3. Se calcula el gradiente medio de `theta0` y `theta1`.
4. Se actualizan ambos parámetros en la dirección que reduce el error.

Las variables se estandarizan durante el entrenamiento para evitar problemas
numéricos causados por la diferencia de escala entre kilómetros y precios. Al
terminar, los parámetros se convierten de nuevo a las unidades originales.

## Pruebas

```bash
python3 -m unittest discover -s tests -v
```

Las pruebas verifican la fórmula de estimación, el aprendizaje de una recta
conocida y el tratamiento de un dataset vacío.

## Posibles mejoras

- Mostrar la evolución de la recta durante el entrenamiento mediante una
  animación.
- Añadir pruebas para archivos CSV y modelos inválidos.

# module2-mlr

## Especificaciones

* Programa uno de los algoritmos vistos en el módulo (o que tu profesor de módulo autorice) sin usar ninguna biblioteca o framework de aprendizaje máquina, ni de estadística avanzada. Lo que se busca es que implementes manualmente el algoritmo, no que importes un algoritmo ya implementado. 
* Prueba tu implementación con un set de datos y realiza algunas predicciones. Las predicciones las puedes correr en consola o las puedes implementar con una interfaz gráfica apoyándote en los visto en otros módulos.

## Librerías utilizadas

* **Pandas** para el manejo del dataset.
* **Matplotlib** para la visualización gráfica.

## Dataset utilizado

**Nombre:** House price prediction

**Kaggle URL:** https://www.kaggle.com/datasets/shree1992/housedata?select=data.csv

## Métrica de desempeño

Para medir el desempeño del modelo se utilizó la función de costo de Gradiente Descendiente para el cálculo del error sobre el subset de entrenamiento y prueba. Los resultados fueron los siguientes:

Error de subset de prueba: 29479486422.36
Error de subset entrenamiento: 25721917027.43

Los errores muestran que hay un nivel de sesgo medio (el valor se ve muy elevado porque está elevado al cuadrado y el precio está en unidades muy grandes, en las predicciones y gráficas se aprecia mejor el desempeño), sin embargo, la varianza analizando el error en los dos subsets es baja.

## Predicciones de prueba

A continuación se muestran algunas entradas, valores esperados y valores obtenidos del modelo.

bedrooms | bathrooms | sqft_living | sqft_lot | floors | view | condition | expected_price | obtained_price
---------|-----------|-------------|----------|--------|------|-----------|----------------|---------------
4.0 | 2.5 |	3180 | 21904 | 2.0 | 3 | 0 | 736500 | 826269.16
3.0 | 2.25 | 2010 | 6000 | 1.0 | 0 | 3 | 570000 | 531978.47
3.0	| 1.75 | 1330 | 7500 |	1.0 |	0 |	3 | 787000 | 347639.41
4.0	| 2.5 |	2370 |	6500 |	2.0 |	0 |	3 | 328000 | 627970.08
5.0	| 2.0 |	1840 |	9240 |	1.0 |	0 |	4 | 435000 | 482350.70

## Nombre del archivo a revisar

main.py

## Correcciones realizadas

Los comentarios de la retroalimentación fueron:
* El Github incluye la descripción de la entrega (ya sea en el readme o en un documento) y contiene todos los elementos solicitados (dataset usado, métrica de desempeño, predicciones de prueba).
* Se muestra la métrica de desempeño del modelo y el desempeño logrado es adecuado.

Se corrigieron de la siguiente manera:
* Se agregó a Github la descripción de la entrega en el README.md.
* Se agregó la métrica de desempeño al código de python.

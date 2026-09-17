---
layout: post
title: "On Building Scorecards"
date: 2026-09-10
categories: [credit-risk, classification, statistics, finance]
---

Bien sea que solicites a un crédito para comprar un nuevo smartphone, para renovar tu casa o comprarla, o para ampliar el cupo de tu tarjeta de crédito o crédito rotativo, la decisión final siempre dependerá de la entidad a la que recurras y en la mayoría de ocasiones, de todas aquellas que componen el intrincado tejido financiero.

La asignación de un puntaje de riesgo es un proceso mediante el cual se establece una magnitud numérica de la susceptibilidad de un cliente apara ser catalogado como malo, esto para una población de clientes potenciales de un negocio o para clientes activos en el mismo.

Los modelos usados para cuantificar dicho riesgo, comúnmente se basan en clasificación binaria, a través de la cuál se establece una probabilidad de que el cliente cumpla con las características más representativas que lo asocien a un mal o buen comportamiento según las reglas de negocio.

Dichas características se estandarizan de forma tabular, creando un medio para acceder a los criterios de puntuación, aún sin un bagaje técnico, democratizando la toma de decisiones y la evaluación efectiva de la "credibilidad" y la calidad de la cartera de un cliente. Por ejemplo[^1],


| Characteristic Name | Attribute | Scorecard Points |
| :--- | :--- | :--- |
| AGE | . -> 23 | 63 |
| AGE | 23 -> 25 | 76 |
| AGE | 25 -> 28 | 79 |
| AGE | 28 -> 34 | 85 |
| AGE | 34 -> 46 | 94 |
| AGE | 46 -> 51 | 103 |
| AGE | 51 -> . | 105 |
| CARDS | "AMERICAN EXPRESS," "VISA OTHERS," "VISA MYBANK," "NO CREDIT CARDS" | 80 |
| CARDS | "CHEQUE CARD," "MASTERCARD/EUROC," "OTHER CREDIT CARD" | 99 |
| EC_CARD | 0 | 86 |
| EC_CARD | 1 | 83 |
| INCOME | . -> 500 | 93 |
| INCOME | 500 -> 1,550 | 81 |
| INCOME | 1,550 -> 1,850 | 75 |
| INCOME | 1,850 -> 2,550 | 80 |
| INCOME | 2,550 -> . | 88 |
| STATUS | "E," "T," "U" | 79 |


El estándar industrial para este tipo de modelos, es la regresión logística, tanto por transparencia en su implementación, así como la interpretabilidad de sus parámetros y los efectos asociados a las covariables (Alineado con Basilea III[^2]). Sin embargo, una adaptación a partir de un análisis SHAP (o de valores Shapley) es posible de tal forma que podamos transitar desde un enfoque paramétrico, a uno no paramétrico.

El objetivo es, pues, presentar una implementación con base a dichas métricas y un modelo basado en árboles y *boosting*: LightGBM (*Light Gradient Boosting Machine*).

Exploraremos en esta entrada: la escogencia de covariables o variables independientes que refejen una alta separabilidad o discriminación entre clientes buenos o malos; el proceso de crear categorías óptimas a partir de covariables cuantitativas mediante un *binning* y, finalmente, como transicionar de los Shapley *values* a los *odds ratio* y posteriormente, usar estos valores para construir un puntaje de la bondad o no, de un cliente. Sin más, comencemos por el principio.

**Def [Bad Client]**: 
A partir de un criterio de negocios, habremos de definir que es un cliente malo: un cliente que paga con 30 días de atraso, un cliente que deja de pagar sus obligaciones, entre otros.

Con base a esta definición y priorización, pasaremos a construir la variable respuesta:

$$
y = \begin{cases} 
1 & \text{if a client is ``bad'' as per business rules} \\
0 & \text{otherwise}
\end{cases}
$$


Luego, a partir de $$y$$, definiremos $$p$$,

$$p = P[y = 1 | \text{a set of independent variables}]$$

y según $$p$$, definimos los *odds* como, 

$$odds = \frac{p}{1-p}$$

Esta definición, corresponde a la magnitud del efecto del evento, vs. la ausencia de este.

Si tuvieramos una variable, $$X_1$$, por ejemplo,

$$
X_1 = \begin{cases} 
1 & \text{if a client is older than 37 years} \\
0 & \text{otherwise}
\end{cases}
$$

Podríamos crear una tabla de contingencia que mida la distribución del fenómeno de interés en función del atributo,


| $$X_1 \setminus Y$$ | $$Y = 1$$ | $$Y = 0$$ | Total |
| :--- | :---: | :---: | :---: |
| **$$X_1 = 1$$** | $$a$$ | $$b$$ | $$a + b$$ |
| **$$X_1 = 0$$** | $$c$$ | $$d$$ | $$c + d$$ |
| **Total** | $$a + c$$ | $$b + d$$ | $$a + b + c + d$$ |

Definiremos ahora, 

$$p_1 = P[Y = 1| X_1 = 1]$$
$$  = \frac{a}{a + b}$$

y, 

$$p_2 = P[Y = 1| X_1 = 0]$$
$$  = \frac{c}{c + d}$$

El *odds* de clientes mayores de 37 años es y aquellos menores de 37 años, respectivamente, son, 

$$odds_1 = \frac{p_1}{1-p_1} = \frac{a/(a + b)}{b/(a + b)} = a/b$$

$$odds_2 = \frac{p_2}{1-p_2} = \frac{c/(c + d)}{d/(c + d)} = c/d$$

Luego, los *odds ratio* son, 

$$OR = \frac{odds_1}{odds_2} = \frac{a/b}{c/d} = \frac{a \times d}{b \times c}$$

Este valor, noa indica la magnitud del efecto de los clientes > 37 vs. los clientes $$\leq$$ 37.

Entonces,

- $$OR = 1$$. Ausencia de asociación. La variable no se asocia con que un cliente sea malo;
- $$OR < 1$$ Asociación negativa. La presencia de la variable genera una disminución en la probabilidad de que el cliente sea malo;
- $$OR \geq 1$$ Asociación positiva. La presencia de variable genera un aumento en la probabilidad de que el cliente sea malo.

## Weight of Evidence & Information Value

Definidos los $$odds$$ y los $$OR$$, podemos introducir el concepto de *Weight of Evidence* y los *Information Values* (WoE & IV en adelante); esta parte se toma y adapta material de Siddiqi (2006)[^1]. El WoE es una medida de la fuerza de cada categoría de una variable cualitativa para separar clientes buenos de aquellos de interés o malos. Se define como la diferencia de la proporción de buenos y malos en cada categoría, esto es,

$$WoE = \Bigg[ ln\Bigg( \frac{Distr. Good}{Distr. Bad } \Bigg)\Bigg] \times 100$$

Por ejemplo, supongamos que tenemos 565,348 clientes, de los cuales, 54,273 son ``malos'' ($$\approx 9.6\%$$). De estos 565,348, un 15% se ubican en el rango de edad  23-26 años, o, aproximadamente, 84,803 clientes.

Los 84,803 clientes en el rango de 23-26 años se dividen luego de la siguiente forma,

|Grupo de Edad| Total Clientes Malos | Total Clientes Buenos | Clientes Malos Grupo | Clientes Buenos Grupo|
| :--- | :--- | :--- | :--- | :--- |
|$$\vdots$$|$$\vdots$$|$$\vdots$$|$$\vdots$$|$$\vdots$$|
|23-26|54,273|511,075|15,265|69,538|
|$$\vdots$$|$$\vdots$$|$$\vdots$$|$$\vdots$$|$$\vdots$$|

Para el rango de 23-26 años, la proporción de buenos respecto al total de buenos es,

$$Distr. Good = \frac{69,538}{511,075} \approx 13.61\%$$

y, la proporción de malos en el mismo rango etario, respecto al total de malos en la población es,

$$Distr. Bad = \frac{15,265}{54,273} \approx 28.13 \%$$

Reemplazando en la fórmula del WoE tenemos,

$$WoE_{23-26} = \Bigg[ ln \Bigg( \frac{0.1361}{0.2813} \Bigg) \Bigg] \times 100$$
$$\approx -72.610$$

Entre más negativo el valor del WoE, más separabilidad se logra por cuenta de los clientes malos respecto a los buenos para esta categoría en particular.

Por otra parte, consideremos $c$ categorías para una variable en particular. El IV, tomado de la teoría de la información (ver[^3]) se mide de la siguiente forma,

$$\sum_{j = 1}^c (Distr. Goods_{j} - Distr. Bads_{j}) \times ln \Bigg(\frac{Distr. Godds_{j}}{Distr Bads_{j}}\Bigg)$$

Para interpretar el IV de una variable, se tiene en cuenta la siguiente regla empírica:

- $$IV < 0.02$$ indica una variable no predictivo;
- $$0.02 \geq IV \leq 0.1$$ indica una variable débilmente predictiva;
- $$0.1 \geq IV \leq 0.3$$ indica una variable medianamente predictiva;
- $$0.3 \geq IV \leq 0.5$$ indica una variable fuértemente predictiva.

Si es mayor a 0.5, se recomienda proceder con cuidado, ya que su poder de predicción es alto, e indica, una alta correlación de la covariable con la definición de clientes malos. 

## Shapley Values and Relationship with *Odds*

Finalmente, tenemos los valores SHAP, anagrama de Shapley Additive Explanations, que consiste en una métrica domada de la teoría de juegos. La definición se toma de Ponce-Bobadilla, et al. (2024)[^4], mientras la fórmula de la aproximación de Molnar (2025)[^5].

Bajo la definición de juegos, se consideran $$N$$ jugadores que trabajarán en conjunto para lograr un propósito en común, esto tras generar una distribución justa para la recompensa de su trabajo, sin llegar a ser igual para todos.

La recompensa se denota por $$\mathcal{V}$$ y cada jugador por $$j$$.El valor Shapley asignado al jugador viene denotado por $$\phi_j$$ que es la distribución justa, de la recompensa, basada en su contribución individual.


La fórmula de $$\phi_j$$ viene dada por,

$$\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} (\mathcal{V}(S \cup \{j\}) - V(S))$$

- $$S$$ es una coalición o combinación de jugadores;
- $$[\mathcal{V}(S \cup \{j\}) - \mathcal{V}(S)]$$ es la contribución de $j$ a la coalición $S$;
- $$\frac{\|S\|! (\|N\| - \|S\| - 1)!}{\|N\|!}$$ es el peso de la contribución marginal;
- Finalmente, la suma considera todas las contribuciones marginales sin $$j$$.

Los pesos son la inversa del número de coaliciones de tamaño $$\|S\|$$ excluyendo al jugador $$j$$.

Los valores Shapley, pueden así, interpretarse como el promedio ponderado de las contribucines marginales de un jugador para todas las posibles coaliciones.

En nuestro caso, el análogo de cada jugador son las variables independientes, las coaliciones, subgrupos de estas y la recompensa, es la predicción llevada a cabo por las $$j$$ variables en una muestra o conjunto de muestras.

Dado que en problemas modernos podemos contar con un número elevado de covariables es necesario contar con una aproximación para $$\phi_j$$.

Consideremos una estimación de interés $$\hat{f}$$ que depende de una fila de observaciones $$\mathbf{x}$$ para cada covariable $$j$$. Para estimar $$\phi_j$$, se puede usar muestreo de Monte Carlo de la siguiente forma:

$$\hat{\phi}_j = \frac{1}{M} \sum_{m = 1}^{M} (\hat{f}(\mathbf{x}_{+j}^{(m)}) - \hat{f}(\mathbf{x}_{-j}^{(m)}))$$

Aquí,

- $$M$$ constituye la totalidad de iteraciones de Monte Carlo y $$m$$, los índices de todas las iteraciones realizadas;

- $$\hat{f}(\mathbf{x}_{+j}^{m})$$: la predicción para $$\mathbf{x}$$ usando valores aleatorios de una fila distinta, $$z$$ para la variable $$j$$;

- $$\hat{f}(\cdot)$$ para $$\mathbf{x}_{-j}^{(m)}$$ es idéntica a $$x_{+j}^{(m)}$$ pero $$\mathcal{x}_{j}^{(m)}$$ también se toma de $$\mathbf{z}$$.

Nuevamente, se remite al lector a Molnar (2025)[^5] para una descripción detallada del algoritmo de aproximación para $$\phi_j$$.

Ahora, bien, es natural pensar, ¿cómo entra a jugar esta teoría en la construcción de mi *score*?

Los modelos de clasificación binaria, no suelen retornar probabilidades crudas, si no, los log-*odds*. Además, los valores SHAP poseen una naturaleza aditiva. De esta forma, podemos descomponer los log-*odds* de la siguiente forma,

$$f(x) = \phi_0 + \sum_{j = 1}^{p} \phi_j(\mathbf{x})$$

con $$p$$ el número de variables y $$\phi_0$$ el valor base o la predicción sin la contribución de ninguna covariable. Como estamos prediciendo log-*odds*, podemos escribir (Lundberg & Lee, 2017[^6]; Nohara, et. al, 2022[^7]),

$$log-odds = \phi_0 + \sum_{j = 1}^{p} \phi_j(\mathbf{x})$$


Y, para una variable $$j$$ con una categoría $$i$$, podemos calcular sus contribuciones de la siguiente forma,

$$\bar{\phi}_{i,j} = \frac{1}{N_j} \sum_{k \in Cat_i} \phi_j (x_k)$$

Finalmente, si queremos construir un score que considere la distribución a nivel de categorías y las variables que las contienen, usamos (Siddiqi, 2006)[^1]:

$$score_j = ln(odds) * \text{Factor} + offset$$

Donde, 

$$\text{Factor} = \frac{\text{PDO}}{ln(2)}$$

y $$\text{PDO}$$ son los puntos necesarios para que un cliente en riesgo, duplique sus $odds$ de ser un cliente bueno.



## Implementation


# Referencias

[^1]: Siddiqi, N.(2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.
                *John Wiley & Sons*. 1st Ed.


[^2]:Basel Committee on Banking Supervision. (2017). *Basel III: Finalising post-crisis reforms*.
                Bank for International Settlements. bis.org

[^3]: Larsen, K. (2015). Data Exploration with Weight of Evidence and Information Value in R. *multithreaded*. URL: https://multithreaded.stitchfix.com/blog/2015/08/13/weight-of-evidence/

[^4]: Ponce-Bobadilla A.V., Schmitt V., Maier C.S., Mensing S., Stodtmann S. (2024). Practical guide to SHAP analysis: Explaining supervised machine learning model predictions in drug development. **Clinical and Translational Science*. doi: 10.1111/cts.70056. 

[^5]: Molnar, C. (2025). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (3rd ed.).christophm.github.io/interpretable-ml-book/

[^6]: Lundberg, S.M. & Lee, S. (2017). A unified approach to interpreting model predictions. *In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17)*. Curran Associates Inc., Red Hook, NY, USA, 4768–4777.

[^7]: Nohara, Y., Matsumoto, K., Soejima, H. & Nakashima, N. (2022). Explanation of machine learning models using shapley additive explanation and application for real data in hospital. *Computer Methods and Programs in Biomedicine*
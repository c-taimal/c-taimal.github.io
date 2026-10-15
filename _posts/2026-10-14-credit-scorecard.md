---
layout: post
title: "Let's build a Scorecard using SHAP values: Part 1"
date: 2026-10-14
categories: [credit-risk, classification, statistics, finance]
---

# Introduction

A common situation many of us have faced when applying for a credit product is to having to establish our creditworthiness. Whether we want to buy a new smartphone, renovate or purchase a home, or increase our credit card limit, the final decision rests with the financial institution we approach and, more broadly, with the institutions that make up the complex financial system.


Creditworthiness is assessed using a quantitative tool known as a $$Scorecard$$, which assigns a numerical value called $$Credit \ Score$$. This value helps assess a customer's credit risk, providing a basis for evaluating both prospective customers (Application Scoring) and existing ones (Behavioral Scoring).

The models used to quantify a customer's (or prospective customer's) behavior are commonly based on binary classification, through which a probability is assessed that the customer meets the most representative attributes associated with good or bad behavior according to the company.

Such attributes are standardized in a tabular format, providing a means to access the point allocation criteria even without a technical background. This enables transparent decision-making and the effective evaluation of creditworthiness and portfolio quality.

As an example, let's consider the following table taken from Siddiqi (2006)[^1]:


| Feature Name | Levels | Scorecard Points |
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

Thus, a customer within the age range 28 -> 34 años, that has MasterCard credit cards, doesn't have an EC_CARD and with a monthly income between 1,850 to 2,550 USD will have the next points allocation:

$$85 + 99 + 83 + 80 + 79 = 426$$

On its own, this number doesn't establish whether a client is good or bad, as this classification is defined by the institution. Thus, scorecards broaden the scope of risk models and allow non-technical workers to classify clients beyond the Good-Bad spectrum

The industry standard for this type of model is Logistic Regression, valued for its implementation transparency, auditability, and the interpretability of its parameters and the statistical effects associated with its covariates (aligned with Basel III[^2]). However, it is possible to make an adaptation via SHAP analysis (or Shapley values) which allows us to transition from a parametric approach to a non-parametric one that is robust against the violation of assumptions such as multicollinearity among covariates, the presence of extreme values, and missing attributes.

The purpose of this entry is thus, to present how to implement a Scorecard using the SHAP values using a *boosting* and *tree-based* model: LightGBM (*Light Gradient Boosting Machine*).

We'll explore in this first part: how to chose covariates or attributes that reflect a better separability between good and bad clientes, how to transition from the SHAP values to the *log-odds* and finally, how to build a Scorecard and allocate points for the goodness or not of a client. Without delay, *let me start from the beginnin'*.


# Preliminaries


## **Def [Bad Client]**: 
If a client is good or bad depends on the business criterion. Usually, a bad client is defined as one that pays late, normally one that pays after 30 days of delay or one that stops paying overall.

Based on this definition and priority, we can construct the target or response variable:


<div>
$$
y = \begin{cases} 
1 & \text{if a client is ``bad'' as per business rules} \\
0 & \text{otherwise}
\end{cases}
$$
</div>

Using $$y$$, now we can define $$p$$,

$$p = P[y = 1 | \text{a set of independent variables}]$$

and through $$p$$, we can define the *odds* as, 

$$odds = \frac{p}{1-p}$$

This definition measures the magnitude of an event against its abscense.

For example if we had a covariate $$X_1$$,

$$
X_1 = \begin{cases} 
1 & \text{if a client is older than 37 years} \\
0 & \text{otherwise}
\end{cases}
$$

We could create a contingency table that measures how the phenomenon is distributed  as a function of the feature,


| $$X_1 \setminus Y$$ | $$Y = 1$$ | $$Y = 0$$ | Total |
| :--- | :---: | :---: | :---: |
| **$$X_1 = 1$$** | $$a$$ | $$b$$ | $$a + b$$ |
| **$$X_1 = 0$$** | $$c$$ | $$d$$ | $$c + d$$ |
| **Total** | $$a + c$$ | $$b + d$$ | $$a + b + c + d$$ |

Now let's define, 

$$p_1 = P[Y = 1| X_1 = 1]$$
$$  = \frac{a}{a + b}$$

and, 

$$p_2 = P[Y = 1| X_1 = 0]$$
$$  = \frac{c}{c + d}$$

The *odds* for clients that are 37 years old or more and those with less than this age are respectively,  

$$odds_1 = \frac{p_1}{1-p_1} = \frac{a/(a + b)}{b/(a + b)} = a/b$$

$$odds_2 = \frac{p_2}{1-p_2} = \frac{c/(c + d)}{d/(c + d)} = c/d$$

Now, the *odds ratio* are, 

$$OR = \frac{odds_1}{odds_2} = \frac{a/b}{c/d} = \frac{a \times d}{b \times c}$$

This value indicates how strong is the effect of being 37 years old against being under 37.

Thus,

- $$OR = 1$$. **Null association**. The covariate doesn't influence whether the client is good or bad;
- $$OR < 1$$ **Negative association**. The covariate presence reduces the probability of a client of being bad;
- $$OR \geq 1$$ **Possitive association**. The covariate presence increases the probability of a client of being bad.

## Weight of Evidence & Information Value

After defining the $odds$ and $OR$, we can introduce the Weight of Evidence and Information Value (WoE and IV hereafter); this material is adapted from Siddiqi (2006)[^1]. The WoE measures the strength of each category of a categorical variable in separating good from bad clients. It is defined as the natural logarithm of the ratio of the distribution of good clients to bad clients, i.e.,

$$WoE = \Bigg[ ln\Bigg( \frac{Distr. Good}{Distr. Bad } \Bigg)\Bigg] \times 100$$

For example, suppose we have 565,348 customers, of whom 54,273 are "bad" (
≈9.6%). Of these 565,348 customers, roughly 
15% fall within the 23–26 age range, representing approximately 84,803 customers.

The 84,803 customers in the age range are then divided as follows,

|Age range| Total of Bad Clients | Total of Good Clients | Bad Clients within Age range| Good Clients within Age range|
| :--- | :--- | :--- | :--- | :--- |
|$$\vdots$$|$$\vdots$$|$$\vdots$$|$$\vdots$$|$$\vdots$$|
|23-26|54,273|511,075|15,265|69,538|
|$$\vdots$$|$$\vdots$$|$$\vdots$$|$$\vdots$$|$$\vdots$$|

For the 23-26 years old range the goods proportions relative to the total goods is,

$$Distr. Good = \frac{69,538}{511,075} \approx 13.61\%$$

and the proportion of bads within the same range relative to the total of bads is,

$$Distr. Bad = \frac{15,265}{54,273} \approx 28.13 \%$$

Replacing for the WoE formula,

$$WoE_{23-26} = \Bigg[ ln \Bigg( \frac{0.1361}{0.2813} \Bigg) \Bigg] \times 100$$
$$\approx -72.610$$

The more negative the WoE a greater separability is achieved between bad and good clients for this particular category.

On the other hand, let's consider $c$ categories for a particular covariate. The IV, a measure adapted from information theory (see[^3]) is defined as follows,

$$\sum_{j = 1}^c (Distr. Goods_{j} - Distr. Bads_{j}) \times ln \Bigg(\frac{Distr. Godds_{j}}{Distr Bads_{j}}\Bigg)$$

For a variable its IV is interpreted as shown below:

- $$IV < 0.02$$ a non-predictive feature;
- $$0.02 \leq IV \leq 0.1$$ a weakly predictive feature;
- $$0.1 \leq IV \leq 0.3$$ a moderate predictive feature;
- $$0.3 \leq IV \leq 0.5$$ a strong predictive feature.

If the IV is greater than 0.5 its recommended to proceed with caution as it could contain how the definition was made for the bad clients (information leakage).

## SHAP Values and their Relationship with *Odds*

Finally we have the SHAP values, anagram of **S**hapley **A**dditive ex**P**lanations which consists of a concept adapted from Game Theory. The definition is the same given by Ponce-Bobadilla, et al. (2024)[^4], while the approximation is the same as Molnar (2025)[^5].

Under a game context, we consider $$N$$ players that work together to achieve a common purpose; also, a fair distribution of the reward is given without it being the same for all players.

The reward is denoted as $$\mathcal{V}$$ and every player as $$j$$. The Shapley value assigned to the player is denoted as $$\phi_j$$ which is the fair distribution of the reawrd based on their individual contribution.


The formula for $$\phi_j$$ is,

$$\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} (\mathcal{V}(S \cup \{j\}) - V(S))$$

- $$S$$ is a coalition or combination of players;
- $$[\mathcal{V}(S \cup \{j\}) - \mathcal{V}(S)]$$ is the contribution of $j$ to coalition $S$;
- $$\frac{\|S\|! (\|N\| - \|S\| - 1)!}{\|N\|!}$$ is the weight of the marginal contribution;
- Finally the sum considers all marginal contributions excluding player $$j$$.

The weights are the inverse of the possible number of coalitions of size $$\|S\|$$ without player $$j$$.

thus, the Shapley values can be considered as the weighted average of marginal contributions of a player for all the possible coalitions.

In our case the analogy to every player are the covariates, the coalition every possible combination of them and the reward the prediction by the $$j$$ covariates in a sample or set of samples.

As the number of covariates is often large, an exact computation is often difficult, thus it's required an approximation for $$\phi_j$$.

Considering an estimation of the response given by $$\hat{f}$$ that depends for a row or observations $$\mathbf{x}$$ with $$j$$ covariates. To estimate $$\phi_j$$, Monte Carlo sampling is used as follows:

$$\hat{\phi}_j = \frac{1}{M} \sum_{m = 1}^{M} (\hat{f}(\mathbf{x}_{+j}^{(m)}) - \hat{f}(\mathbf{x}_{-j}^{(m)}))$$

Here,

- $$M$$ are all the Monte Carlo iterations and $$m$$ indexes every possible iteration;

- $$\hat{f}(\mathbf{x}_{+j}^{m})$$: is the prediction for the row $$\mathbf{x}$$ but using values from a different row: $$z$$ for the variable $$j$$;

- $$\hat{f}(\cdot)$$ for $$\mathbf{x}_{-j}^{(m)}$$ is identical to $$x_{+j}^{(m)}$$ but $$\mathcal{x}_{j}^{(m)}$$ is also taken from $$\mathbf{z}$$.

Again, the reader is directed to Molnar (2025)[^5] for a detailed description of the algorithm to approximate $$\phi_j$$.

It's completely normal to think, how is this theory related to the construction of my *score*?

Well, the binary classification models, whether it be parametric or non-parametric do not return probabilites but the *log-odds*. On the other hand, as the SHAP values have an additive nature we can decompose its prediction like this,

$$f(\mathbf{x}) = \phi_0 + \sum_{j = 1}^{p} \phi_j(\mathbf{x})$$

with $$p$$ the total of covariates and $$\phi_0$$ the base value or the default prediction without any covariate. As we are predicting *log-odds* we can write (Lundberg & Lee, 2017[^6]; Nohara, et. al, 2022[^7]),

$$log-odds = \phi_0 + \sum_{j = 1}^{p} \phi_j(\mathbf{x})$$


and for a covariate $$j$$ with a category  $$k$$, we can compute its contributions in the following way,

$$\bar{\phi}_{j, k} = \frac{1}{N_k} \sum_{h \in Cat_k} \phi_j (x_{j, h})$$

## From SHAP Values to Points to Score

If we want to construct a Score that considers the distributions per category of the *log-odds* and the variables associated to such category, we need to make the following scaling process (Siddiqi, 2006)[^1]:

$$score = ln(odds) * \text{Factor} + Offset$$

With, 

$$\text{Factor} = \frac{\text{PDO}}{ln(2)}$$

From now, we'll call $$B = \text{Factor}$$. $$\text{PDO}$$ is a scaling factor that controls how soft or abrupt are the effects of doubling the Good:Bad $$odds$$. This factor is related to another scaling element that is implicit as we'll see: the $Target \ Odds$, which as its name suggests reflects how many good clients are by every bad and finally, we have the $$Offset$$; the $$Offset$$ allows us to scale the *Score* around a reference value called $$Target \ Score$$ ($$Score_0$$). Now, let's suppose we have a referenc score $$Score_0$$, then:

$$Score_0 = Offset + B ln (Odds_0)$$

$$ \Longrightarrow Offset = Score_0 - {B \times ln (Odds_0)}$$


$$Odds_0$$ is the same Target odds and in the industry, a common value is 50: 1, that is, for very bad client there are 50 bads, or,

$$p(bad) = \frac{1}{1+50} = \frac{1}{51} \approx 19.60\%$$

If we would set $Target \ Odds = 50:1$ and we had that a client increases its PDO (en 20 unidades), the Good:Bad odds increase as well in, approximately, the double or 100:1.

Using the previos information we can now construct our score $$Score$$ using the SHAP.

As we know, the SHAP values are additive and in a *log-odds* scale. For a client or row $$\mathbf{x}$$ its prediction of being bad is given by,

$$ln(Odds(\mathbf{x})) = \phi_0 + \sum_{j = 1}^p \phi_j(x_j)$$

Replacing in our scaling formula we have,

$$Score = Offset + B \Big[ \phi_0 + \sum_{j = 1}^p \phi_j(x_j) \Big]$$

$$\Longrightarrow Score = \Big[ Offset + B \phi_0 \Big] + \sum_{j = 1}^p B \phi_j(x_j)$$

$$Offset + B \phi_0 $$ can be called *Base Score*, with $$Offset$$ already known and in function of $$Score_0$$, $$B$$ and $$Odds_0$$. Thus,

$$Base \ Score = Score_0 - B ln(Odds_0) + B \phi_0$$
$$ \Longrightarrow Puntos \ Base = Score_0 + B [\phi_0 - ln(Odds_0)]$$

Finally,

$$Score = Score_0 + B [\phi_0  - ln(Odds_0)] +   \sum_{j = 1}^p B \phi_j(x_j)$$


In this case, $$\phi_0$$ is similar to $$\beta_0$$ in the logistic regression and measures the average contribution for all the covariables, again, in the log-odds scale. For every variable $$j$$, 

$$\text{Points}_j (\mathbf{x}) = - B \phi_j(x_j)$$

And for every client $$\mathbf{x}$$  their score is given by,

$$Score(\mathbf{x}) = Score_{base} + \sum_{j = 1}^p Points_j(x_j)$$

with,

$$Score_{base} = Score_0 + B [\phi_0  - ln(Odds_0)]$$.

# Conclusion

So far, we dilucidated how the $$OR$$ and the SHAP values are connected, and, as well we could create a workflow to make an implementation. Thus, to create a scorecard we cannot only rely on parametrical procedures that are subjected to all sort of assumptions, but also generalize to a modern machine learning setting which is robust and can scalete in a better way. 

Stay tuned for the second part where we'll implement this methodology in Python and allow me to walk you through this.

Thanks for your time :).


# Bibliography

[^1]: Siddiqi, N.(2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.
                *John Wiley & Sons*. 1st Ed.


[^2]:Basel Committee on Banking Supervision. (2017). *Basel III: Finalising post-crisis reforms*.
                Bank for International Settlements. bis.org

[^3]: Larsen, K. (2015). Data Exploration with Weight of Evidence and Information Value in R. *multithreaded*. URL: https://multithreaded.stitchfix.com/blog/2015/08/13/weight-of-evidence/

[^4]: Ponce-Bobadilla A.V., Schmitt V., Maier C.S., Mensing S., Stodtmann S. (2024). Practical guide to SHAP analysis: Explaining supervised machine learning model predictions in drug development. **Clinical and Translational Science*. doi: 10.1111/cts.70056. 

[^5]: Molnar, C. (2025). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (3rd ed.).christophm.github.io/interpretable-ml-book/

[^6]: Lundberg, S.M. & Lee, S. (2017). A unified approach to interpreting model predictions. *In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17)*. Curran Associates Inc., Red Hook, NY, USA, 4768–4777.

[^7]: Nohara, Y., Matsumoto, K., Soejima, H. & Nakashima, N. (2022). Explanation of machine learning models using shapley additive explanation and application for real data in hospital. *Computer Methods and Programs in Biomedicine*

[^8]: Navas-Palencia, G. (2022). Optimal binning: mathematical programming formulation. Arxiv preprint: https://arxiv.org/abs/2001.08025
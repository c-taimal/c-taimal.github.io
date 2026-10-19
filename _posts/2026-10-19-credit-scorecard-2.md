---
layout: post
title: "Let's build a Scorecard using SHAP values: Part 2"
date: 2026-10-19
categories: [credit-risk, classification, statistics, finance]
---

# **Implementation**

Some aspects of this exercise:

- We'll use th **All Lending Club Data** available on [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club);


- Though feature engineering could be performed this is not the focus of this exercise, but rather, explain how to transition from SHAP values to a scorecard.

**Methods:** LightGBM, Logistic Regression, SHAP Values, $t$-**S**tochastic **N**eighbor **E**mbedding ($t$-SNE).

**Python Libraries:** $$\texttt{optbinning}, \texttt{lightgbm}, \texttt{scikit-learn}, \texttt{pandas}, \texttt{numpy}, \texttt{matplotlib}, \texttt{seaborn}$$

## **Libraries**

Let's start by loading the libraries that are assumed to be already installed, as well as the dataset. The notebook with the implementation and the requirements are located in [this gist](https://gist.github.com/c-taimal/bd9344eceb37c1dce7a6df61ebcb9193).

We'll use the Python version 3.14.7.


```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
```


## **Data Loading**

As mentioned previously, we'll be using the Loan Club Lending dataset, most specifically, the accepted clients. The rejected ones can be used for a Rejection Inference. On the other hand, only features with under 30% of missing values will be taking into account,

```python
# url: https://www.kaggle.com/datasets/wordsforthewise/lending-club/data
files_ls = os.listdir()
files_ls = [f for f in files_ls if '.csv' in f]

accepted_df = pd.read_csv(files_ls[-1])


# Deleting features with more than 30% of missing values

columns_to_drop = (accepted_df.columns)[accepted_df.isna().mean(axis = 0)>0.3]
accepted_df = accepted_df.drop(columns = columns_to_drop)

# Deleting id column
accepted_df = accepted_df.drop(['id'], axis = 1)
```

To define the target or response variable, our analysis will focus only on two type of clients: those who Fully Paid and those who defaulted or marked as Charged-off.


```python
accepted_df = accepted_df.loc[accepted_df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
accepted_df = accepted_df.reset_index(drop = True)

accepted_df['loan_status'] = accepted_df['loan_status'].replace({'Fully Paid': 0, 'Charged Off': 1})
```

## **Time-based Splitting and Variable Pre-selection**

To implement our classification models, we must consider the clients have an associated date of issue for the credits. This date could implicitly influence our clients behavior through association with macroeconomic proxy and credit-related variables. The issue date variable is encoded as `issue_d` and goes from 2007 to 2018.

<img src="https://c-taimal.github.io/images/issue_date_ts.png" style="display: block; margin: auto;" />

Before 2014 there is an increase in the number of issued credits with a maximum volume of clients reached between 2014-2017 with a decrease after 2017 until 2018. While random data splitting for training, validation, and testing could be performed, variables measured after the time window could influence our results and create a leakage; to avoid such behavior, we start by using `issue_d` as an index for our data and to sort them chronologically. Next, we'll use this order to split our data into the three common subsets,

```python
from datetime import *
accepted_df['issue_d'] = pd.to_datetime(accepted_df['issue_d'], format='%b-%Y')
accepted_df = accepted_df.sort_values('issue_d').reset_index(drop = True)
```
As we have a large enough dataset, we split the data into a 70% for training, the next (chronologically) 15% are for validation or calibrating the classification threshold and finally, the 15% left are used for prediction or testing.

```python
n = len(accepted_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
```

Now we divide our data into covariates and target variable:

```python
covariables = accepted_df.drop(columns = ['loan_status'])
target = accepted_df['loan_status']

train_df = covariables.iloc[:train_end]
y_train = target.iloc[:train_end]

val_df = covariables.iloc[train_end:val_end]
y_val = target.iloc[train_end:val_end]

test_df = covariables.iloc[val_end:]
y_test = target.iloc[val_end:]
```

<img src="https://c-taimal.github.io/images/splitting_scheme.png" style="display: block; margin: auto;" />

After splitting we'll work validating and selecting variables on the training set. For this, we first define what features are qualitative and which ones are quantitative,

```python
categoricas = (train_df.columns)[(train_df.dtypes != np.dtype('float64') ) & (train_df.dtypes != np.dtype('int64'))]
numericas   = (train_df.columns)[(train_df.dtypes == np.dtype('float64') ) | (train_df.dtypes == np.dtype('int64'))]
```

To drop the `issue_d` variable, let's create one feature that measures how many days have passed since the loan disbursement,

```python
train_df['credit_history_length_days'] = pd.to_datetime(train_df['issue_d'], format='%b-%Y') - pd.to_datetime(train_df['earliest_cr_line'], format='%b-%Y')
train_df['credit_history_length_days']  = (train_df['credit_history_length_days']).dt.days

val_df['credit_history_length_days'] = pd.to_datetime(val_df['issue_d'], format='%b-%Y') - pd.to_datetime(val_df['earliest_cr_line'], format='%b-%Y')
val_df['credit_history_length_days']  = (val_df['credit_history_length_days']).dt.days

test_df['credit_history_length_days'] = pd.to_datetime(test_df['issue_d'], format='%b-%Y') - pd.to_datetime(test_df['earliest_cr_line'], format='%b-%Y')
test_df['credit_history_length_days']  = (test_df['credit_history_length_days']).dt.days
```

After that, we update our variable list omitting additional variables that are date-related and non qualitative,

```python
# Let's update our list of variables by type

categoricas = categoricas.drop(['issue_d', 'last_pymnt_d', 'earliest_cr_line', 'last_credit_pull_d'])
numericas = list(numericas) + ['credit_history_length_days']
```

Now, we can implement the features pre-selection. On the qualitative variables the Information Value will be used, while the quantitative ones will be chosen through the Kolmogorov-Smirnov test.

To select the qualitative features, a threshold of a minimum $IV$ of 0.1 is set while a ceiling of 0.5 is defined. We want our covariates to be at least, moderately predictive without them to leak information or redundant in how the response is defined.

Let's remember how the Kolmogorov-Smirnov test works for two samples $$m_1$$ y $$m_2$$. This test wants to measure whether the Empirical Cumulative Distribution Function (ECDF) of both samples is equal or not:

$$H_0: \text{Both samples come from the same distribution} \ (F_{m_1}(x) = F_{m_2}(x))$$
$$vs.$$
$$H_a: \text{Both samples come from different distributions.}$$

If they are, we must reject the null hypothesis, $$H_0$$ which is done when the $$p$$-value is less than the significance level, $$\alpha = 0.05$$. Using this criterion the variables that distinguish between defaulters and non defaulters are the ones chosen for the modeling process; in addition, the distance between the ECDFs is measured with the $$D$$-statistic. Thus, the closer is $$D$$ to 1, the more different the populations will be. A threshold of $$D > 0.1$$ is set to choose quantitative variables.


### **Qualitative Variables Pre-selection**

Starting with the qualitative variables, let's compute the IV between each one and the target variable, `loan_status`:

```python

def calc_iv(df, feature, target, pr=0):

    # Taken from: https://www.kaggle.com/code/podsyp/information-value-iv-weight-of-evidence-woe

    lst = []

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature, val, df[df[feature] == val].count()[feature], df[(df[feature] == val) & (df[target] == 1)].count()[feature]])

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Bad'])
    data = data[data['Bad'] > 0]

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = (data['Distribution Good']/data['Distribution Bad']).replace(0, 0.0001).apply(np.log)
    data['IV'] = (data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])).sum()

    data = data.sort_values(by=['Variable', 'Value'], ascending=True)

    if pr == 1:
        print(data)

    return data['IV'].values[0]
```

We use the code created by [podsyp](https://www.kaggle.com/podsyp) for this implementation. After that, we found `term` and `grade` with an IV of 0.26 and 0.47, respectively, big enough to conclude there is predictability without a leakage.

```python
iv_ls = []
for c in categoricas:
    temp_df = train_df_sample.dropna()
    iv_ls.append(calc_iv(temp_df, c, 'default'))

iv_df = pd.DataFrame({'Variable': categoricas, 'IV': iv_ls})
iv_df = iv_df.loc[(iv_df['IV'] > 0.1) & (iv_df['IV'] < 0.5)]
```

|Feature| IV|
| :--- | :--- |
|`term`|0.262|
|`grade`|0.474|

### **Quantitative Variables Pre-selection**

To compute the Kolmogorov-Smirnov test for the quantitative variables, we'll use the  `kstest` function from the `stats` module of `scipy`:

```python
from scipy.stats import kstest

p_values_ls = []
d_statistic_ls = []

for var in numericas:

    group_1 = train_df.loc[y_train == 0, var].dropna()
    group_2 = train_df.loc[y_train == 1, var].dropna()
    results = kstest(group_1, group_2)

    p_values_ls.append(results.pvalue)
    d_statistic_ls.append(results.statistic)

ks_df = pd.DataFrame({'variables': numericas, 'p_values': p_values_ls, 'd_statistic': d_statistic_ls})
ks_df = ks_df.loc[((ks_df['p_values'] < 0.05) & (ks_df['d_statistic'] > 0.1))]
```
We thus obtain 15 quantitative variables marked as significant by the test:

`int_rate`, `dti`, `fico_range_high`, `fico_range_low`, `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_late_fee`, `recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`, `last_fico_range_high`,  `last_fico_range_low`, `acc_open_past_24mths`,  `num_tl_op_past_12m`.

At a global level we have the following variables to train our model,

| Feature | Definition |
| :--- | :--- |
|`term` |	The number of payments on the loan. Values are in months and can be either 36 or 60.|
|`grade` |	LC assigned loan grade.|
|`int_rate` |	Interest Rate on the loan.|
|`dti` |	A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower's self-reported monthly income.|
|`fico_range_high` |	The upper boundary range the borrower's FICO at loan origination belongs to.|
|`fico_range_low`	| The lower boundary range the borrower's FICO at loan origination belongs to.|
|`total_pymnt` |	Payments received to date for total amount funded.|
|`total_pymnt_inv` |	Payments received to date for portion of total amount funded by investors.|
|`total_rec_prncp` |	Principal received to date.|
|`total_rec_late_fee` |	Late fees received to date.|
|`recoveries` |	post charge off gross recovery.|
|`collection_recovery_fee` |	post charge off collection fee.|
|`last_pymnt_amnt` |	Last total payment amount received.|
|`last_fico_range_high` |	The upper boundary range the borrower's last FICO pulled belongs to.|
|`last_fico_range_low`  |	The lower boundary range the borrower's last FICO pulled belongs to.|
|`acc_open_past_24mths` |	Number of trades opened in past 24 months.|
|`num_tl_op_past_12m`	| Number of accounts opened in past 12 months.|

Let's now explore how these variables behave.

## **Exploratory Data Analysis**

**Qualitative Variables**

<img src="https://c-taimal.github.io/images/qualitatives_eda.png" style="display: block; margin: auto;" />

As we said, we only have as qualitative features the internal grading by the company and the term of the loan in months. For the grading, in an increasing manner (G->F->E->D->C->B->A, i.e., A is better and G is worst) we observe the charged-off clients concentrate in the lower leverls, mostly around the mid grades: C, D, and E, while fully paid clients or not charged-off mostly often, have a grade of A or B. 

On the other hand, the fully paid clients tend to be on the 36 months of term, with around of 80% of records in this category, while for the ones charged-off, the clients often exhibit a larger concetration in the 60 months term, with around 44%. Besides that, the distribution of charged-off clients is approximately the same between the two possible terms.

Thus, these two variables can give us a clear distinction between charged and non charged off clients reafirming the election made through the IV. Finally we decide to check if there's any suspicious association between the features, so we use the Cramér's V statistic which goes from 0 for null association to 1 for complete association between categorical variables.

This statistic is defined as,

$$V = \sqrt{\frac{\chi^2}{n \cdot \text{min}\{r-1, c-1\} }}$$

with $$n$$ the total of elements we have after making a contingency table between our two categorical variables of interest, $$r$$ the number of categories for the first feature, $$c$$ the number of categories for the second one, and $$\chi^2$$ the statistic from the test of the same name.

```python
cor_cramer = np.empty([2, 2])
from scipy.stats import chi2_contingency

def cramer_v (var_1, var_2):

    """
    Computes Cramér's V for association between two categorical features.
    Only needs feature 1 or var_1 and feature 2 or var_2.
    r, k: number of categories per feature,
    n: number of all elements in the cross table.
    """

    cross_table_v = pd.crosstab(train_df[var_1], train_df[var_2])
    chi_2 = chi2_contingency(cross_table_v, correction = False).statistic
    r, k = cross_table_v.shape
    n = cross_table_v.sum().sum()

    min_dim = np.min([r-1, k-1])

    cramer_statistic = np.sqrt(chi_2/(min_dim * n))

    return cramer_statistic

cor_cramer[0,1] = cramer_v('term', 'grade')
cor_cramer[1, 0] = cramer_v('term', 'grade')

cor_cramer[0, 0] = cramer_v('term', 'term')
cor_cramer[1, 1] = cramer_v('grade', 'grade')
sns.heatmap(cor_cramer * 100, cmap = 'coolwarm', annot = True, fmt = '.2f')
```
<img src="https://c-taimal.github.io/images/corr_qualitatives.png" style="display: block; margin: auto;" />

Now, as the Crámer's V is not over 50% we can conclude the association between these two features is moderate.

**Quantitatives**

Now, let's proceed to check on how the quantitative covariates behave. For this, we plot the Empirical Cumulative Distribution Functions disaggregated by fully paid and charged-off clients.

<img src="https://c-taimal.github.io/images/quantitatives_eda.png" style="display: block; margin: auto;" />

Analyzing the values for the minimum and maximum obtained FICO scores, overall, the Lending Club clients have a moderate behavior in the financial system such that the range usually starts around 650, having this score a minimum value of 300. Nevertheless, as it's expected non defaulters or non charged-off clients usually have a higher score than those who have fully paid.

On the other hand,  `dti` shows a global range that goes from 0 to 1000 with a minimum distinction between the curves for charged and non-charged off clients. The interest rates, are usually larger in charged-off clients with this variable, that could be considered as proxies of the purchasing power: non charged-off clients usually pay larger fees per month, nonetheless, naturally, when it comes to debt and delayed payments, charged-off clients pay larger values.

Finally we have variables that measure how the clients behaves in terms of the number of products in the financial system. Charged-off clients usually look for other alternatives as their fees increase with Lending Club.

Let's check for additional leakage using the biserial-point correlation,

```python
correlacion_ls = []
for var in list(ks_df['variables'].values):

    correlation_stat = pointbiserialr(plot_data.dropna(subset = var)[var],
                                      plot_data.dropna(subset = var)['target']).statistic
    correlacion_ls.append(correlation_stat)
```

|Feature|	Correlation to `loan_status`|
|:-----|:-----|
|`recoveries`|	0.515|
|`collection_recovery_fee`|	0.482|
|`int_rate`|	0.258|
|`total_rec_late_fee`|	0.128|
|`acc_open_past_24mths`|	0.117|
|`dti`|	0.109|
|`num_tl_op_past_12m`|	0.101|
|`fico_range_high`|	-0.128|
|`fico_range_low`|	-0.128|
|`total_pymnt_inv`|	-0.280|
|`total_pymnt`|	-0.281|
|`last_pymnt_amnt`|	-0.309|
|`total_rec_prncp`|	-0.415|
|`last_fico_range_low`|	-0.543|
|`last_fico_range_high`|	-0.621|

As FICO scores can have additional and outside information from our time window, we decide not to include them in the analysis.

We also check for variables with high correlation between them (over 70%). Even though the LightGBM model can handle this kind of relationships easily, and it's robust to such behaviors, we decide to omit them in order to keep a parsimonious model.

<img src="https://c-taimal.github.io/images/corr_quantitatives.png" style="display: block; margin: auto;" />

```python
ks_df.loc[ks_df['variables'].isin(['recoveries', 'collection_recovery_fee',
                                  'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp'])]
```

|Variable	|$$p$$-values	|$$D$$-statistic|
|:-----|:-----|:-----|
|`total_pymnt`|	0.000|	0.344|
|`total_pymnt_inv`|	0.000|	0.344|
|`total_rec_prncp`|	0.000|	0.559|
|`recoveries`|	0.000|	0.751|
|`collection_recovery_fee`|	0.000|	0.705|

ultimately, our criterion to keep one variable or another lies in the size of the $D$ statistic so we decide to drop `collection_recovery_fee`, `total_pymnt` and `total_pymnt_inv`.

```python
to_drop_numericas = ['collection_recovery_fee', 'total_pymnt', 'total_pymnt_inv']
numericas_final_ls = [var for var in numericas_final_ls if var not in to_drop_numericas]
```

Thus and with the results obtained for the qualitative features, we can profile the clients as follows:

- **Charged-Off:** internal lower grade with a high level of fees for delayed payments. A large number of external products opened products as well higher interest rates and a larger term to pay the loans.

- **Fully paid:** a higher internal grade with larger fees without having to pay for delayed payments,  less products opened in external institutions and overall, a smaller payment term and lower interest rates.

## **Model Fitting**

### **Binning Quantitative Features**

We already chose the numerical variables with the highest predictive power yet, the scoring procedure both parametric and non-parametric works with binned or encoded variables.

To accelerate this process we'll use the `optbinning` library which roughly speaking, solves a Convex Mixed-Integer Programming problem to find the best binning, both in number of bins and its cut points, to maximize a given metric that better separate our response, be it: Information Value, Gini or Log-Likelihood. A rigorous description is given in Navas-Palencia (2022)[^1].

By default we define a maximum number of bins equal to 5 as we don't want the final scorecard to be of difficult interpretation and a minimum of 2. We look for a maximization of the IV.

```python
from optbinning import OptimalBinning

for var in numericas_final_ls:


    x = train_df[var]
    y = y_train
    optb = OptimalBinning(name = var, dtype="numerical", solver="cp", min_n_bins=2, max_n_bins=5)
    x_binned = optb.fit_transform(x, y, metric = 'bins')
    x_val_binned  = optb.transform(val_df[var], metric = 'bins')
    x_test_binned = optb.transform(test_df[var], metric = 'bins')

    train_df[var + '_binned'] = x_binned
    val_df[var + '_binned']   = x_val_binned
    test_df[var + '_binned']  = x_test_binned

list_variables_model = list(iv_df['Variable'].values) + [var for var in train_df.columns if '_binned' in var]
```

The binned variables will have the suffix `_binned` and with `grade` and `term` we define all our features as qualitatives or categorical,

```python
train_df[list_variables_model] = train_df[list_variables_model].astype('category')
val_df[list_variables_model]   = val_df[list_variables_model].astype('category')
test_df[list_variables_model]  = test_df[list_variables_model].astype('category')
```
and now we can proceed with our traning.

### **Model Training**

Let's define an ensemble with 1000 trees and the other options are left in their default settings. After training our model we compute the default probabilities on the validation set,

```python
import lightgbm as lgb
lgb_classifier = lgb.LGBMClassifier(n_estimators = 1000, verbose = -1)
lgb_classifier.fit(train_df[list_variables_model], y_train)

y_probs_validation = lgb_classifier.predict_proba(val_df[list_variables_model])[:, 1]
```
Now, we define a set of possible thresholds going from 0.05 to 0.8,

```python
thresholds = np.linspace(0.05, 0.8, 20)


recall_ls = [recall_score(y_val.astype('int'), np.where(y_probs_validation > t, 1, 0)) for t in thresholds]
f1_ls = [f1_score(y_val.astype('int'), np.where(y_probs_validation > t, 1, 0)) for t in thresholds]
pr_auc = [average_precision_score(y_val.astype('int'), np.where(y_probs_validation > t, 1, 0)) for t in thresholds]

print(f'Best Recall: {np.max(recall_ls)} | Best PR-AUC: {np.max(pr_auc)} | Best F1: {np.max(f1_ls)}')
```
Using these thresholds we have the following values for the validation set metrics, Recall, the PR-AUC (**PR**ecision curve **A**rea **U**nder **C**urve) and the $F_1$ score:

`Best Recall: 0.9825425425425426 | Best PR-AUC: 0.7667696566380039 | Best F1: 0.8630087910270896`

This recall seems suspicious. To further see if we have a leakage, we fit one model per variable and check its ROC-AUC (Area Under Curve for Receiver Operational Curve) to see if the variable can explain the whole default phenomenon on its own,

```python
from sklearn.metrics import roc_auc_score
roc_auc_ls = []

for var in list_variables_model:

    lgb_classifier_var = lgb.LGBMClassifier(n_estimators = 1000, verbose = -1)
    lgb_classifier_var.fit(train_df[[var]], y_train)
    y_val_pred = lgb_classifier_var.predict(val_df[[var]])
    roc_auc_ls.append(roc_auc_score(y_val, y_val_pred))

leakage_df = pd.DataFrame({'variable': list_variables_model,
                              'roc_auc': roc_auc_ls})
                        
leakage_df
```

|Feature|ROC-AUC|
|:-----|:-----|
|`term`|	0.500|
|`grade`|	0.500|
|`int_rate_binned`|	0.500|
|`dti_binned`|	0.500|
|`total_rec_prncp_binned`|	0.813|
|`last_pymnt_amnt_binned`|	0.500|
|`acc_open_past_24mths_binned`|	0.500|
|`num_tl_op_past_12m_binned`|	0.500|

We remove `total_rec_prncp_binned` as it could create leakage into our data, because total principal received is heavily tied to whether a loan reached full maturity or default post-facto, making it a target leak. After that, we proceed to a second training of our model.

```python
list_variables_model = [var for var in list_variables_model if var != 'total_rec_prncp_binned']

lgb_classifier = lgb.LGBMClassifier(n_estimators = 1000, verbose = -1)
lgb_classifier.fit(train_df[list_variables_model], y_train)
y_probs_validation = lgb_classifier.predict_proba(val_df[list_variables_model])[:, 1]
```
Again, let's compute our metrics and now plot them against a set of possible thresholds,

```python
recall_ls = [recall_score(y_val, np.where(y_probs_validation > t, 1, 0)) for t in thresholds]
f1_ls = [f1_score(y_val, np.where(y_probs_validation > t, 1, 0)) for t in thresholds]
pr_auc = [average_precision_score(y_val, np.where(y_probs_validation > t, 1, 0)) for t in thresholds]

sns.lineplot(x = thresholds, y = recall_ls, label = 'Recall')
sns.scatterplot(x = thresholds, y = recall_ls)

sns.lineplot(x = thresholds, y = pr_auc, label = 'Precision AUC')
sns.scatterplot(x = thresholds, y = pr_auc)

sns.lineplot(x = thresholds, y = f1_ls, label = 'F-1')
sns.scatterplot(x = thresholds, y = f1_ls)
```

<img src="https://c-taimal.github.io/images/thresholds_metrics.png" style="display: block; margin: auto;" />

Around 25%  (number 5 threshold) we get the best behavior and balance between metrics, so we decide this value as the one that will mark a client as charged-off or not.

```python
(recall_ls[5], pr_auc[5], f1_ls[5])
```
Using this model and with the threshold already tuned, we test it in our unseen subset.

### **Using the Testing Set**

Now let's see how the model behaves on unseen data with the threshold calibrated and after fitting our initial model,

```python
y_test_probs = lgb_classifier.predict_proba(test_df[list_variables_model])[:, 1]
y_test_preds = np.where(y_test_probs > thresholds[5], 1, 0)
print(f'Recall Test: {recall_score(y_test, y_test_preds)} | PR-AUC Test: {average_precision_score(y_test, y_test_preds)} | Best F1: {f1_score(y_test, y_test_preds)}')
```

`Recall Test: 0.7908489060005635 | PR-AUC Test: 0.6883432952753561 | Best F1: 0.8025300171526587`

The correctly classified elements in this set is high without overfitting and furthermore, the number of incorrectly classified non charged-off clients is also low as per the PR-AUC. Let's now see how every group is classified,

```python
cross_tab_test = pd.crosstab(y_test, y_test_preds, normalize = 'index') * 100
sns.heatmap(cross_tab_test, cmap = 'coolwarm', annot = True, fmt = '.2f')
```

Our results can be generalized to our testing set, the one the model hasn't interacted yet until this point. Furthermore, this splitting was made time-wise allowing us to remove any leakage from future observations.

Until this point, we have trained a model that can discriminate between defaulters and non-defaulters. Our next step is to codify these rules in a tabular manner. To do so, we'll rely on the SHAP values as we explained in part one.

### **Building our Non-parametric Scorecard**

To compute the SHAP values we use the library with the same name,`shap`. As we'll have to compute a SHAP per row or client we use a lower sample from the initial  ($$\approx$$) 200,000. Thus, we sample 10,000 clients.

```python
import shap

x_shap = test_df[list_variables_model].copy()
x_shap = x_shap.sample(n = 10000, random_state = 2246).astype('category')
```

using `x_shap` we compute the contributions made by every variable to predict every client; let's remember our points are computed in term of log-odds, thus, the `model_output` is set to `raw`.


```python
explainer = shap.TreeExplainer(lgb_classifier, model_output = 'raw')

shap_values = explainer(x_shap)
```

Now to further enforce interpretability we'll work only with the five more important features,

```python
shap_importance = np.abs(shap_values.values).mean(axis=0)

feature_importance = pd.Series(shap_importance, index=x_shap.columns).sort_values(ascending=False)

top_5_features = feature_importance.head(5).index.tolist()
```

These are: `last_pymnt_amnt_binned`, `int_rate_binned`, `grade`, `term` and `acc_open_past_24mths`. Using the `shap_values` and the index of the most important features we bring $$\phi_0$$ and $$\phi_j, \ j = 1, 2, 3, 4, 5$$; $$\phi_0$$ is computed for all variables, while there'll be a $$\phi_j, j = 1, 2, 3, 4, 5$$ for every client (i.e., 50,000 SHAP or a matrix of size $$10,000 \times 5$$):

```python
phi_0 = shap_values.base_values[0]

top_5_indices = [x_shap.columns.get_loc(col) for col in top_5_features]
phi_j_top5 = shap_values.values[:, top_5_indices]
```

After we have defined these values we'll use a custom function to scale the points, using as parameters the $$\texttt{PdO}$$, $$Score_0$$ and $$Odds_0$$ we choose:

```python
scorecard_tables = {}

def scoring_shap_values(score_0: float, odds_0: float,
                        pdo: float):

    """
    Parameters:
    - score_0: or the 'average' score we expect our clients to have;
    - odds_0: how many good clients there are for every bad one;
    - pdo: points to double the odds. How many points must increase or decrease a client to change its current score.
    """

    # The pdo must be negative as the odds here reflect probability of defaulting. Otherwise it would be positive.
    b = -pdo / np.log(2)

    # Similarly the odds are transformed to represent the odds of defaulting
    score_base = score_0 + b * (phi_0 - np.log(1 / odds_0))

    for col in top_5_features:

        col_idx = list_variables_model.index(col)

        df_shap = pd.DataFrame({
            "category": x_shap[col],
            "shap": shap_values.values[:, col_idx]
        })

        category_summary = df_shap.groupby("category", observed=False)["shap"].mean().reset_index()
        category_summary['points'] = np.round(b * category_summary['shap'])
        
        scorecard_tables[col] = category_summary


    variables_score = x_shap[top_5_features].copy()

    for col in top_5_features:
        table = scorecard_tables[col]
        points_map = table.set_index('category')['points'].to_dict()
        variables_score[col] = variables_score[col].map(points_map)


    return score_base, variables_score
```

We observe the $$\text{PdO}$$ must reverse their polarity as we are modeling a negative event, i.e., the $$\texttt{PdO}$$ reduce the score if there's an increase in the log-odds of being charged-off. 

In a similar way, we revert the order of the $$Odds_0$$ so the charged-off probability is better reflected.

Finally, our function returns the points codified per bin for every covariate after setting the defaults values used by the industry,

- $$Score_0 = 600$$;
- $$Odds_0 = 50 (50:1)$$;
- $$\text{PdO} = 20$$.

```python
score_base, scores_df = scoring_shap_values(score_0 = 600, odds_0 = 50, pdo = 20)
scores_df = scores_df.reset_index(drop = True)
scores_df['score'] = (score_base + scores_df.sum(axis = 1))
```

We also check for scores under 300 and additionally, we retrieve the `loan_status` label we used to model the score,

```python
print(f'Customers with score < 300: {scores_df.loc[scores_df['score'] < 300, 'score'].shape[0]}')
scores_df.loc[scores_df['score'] < 300, 'score'] = 300
scores_df['default'] = y_test.loc[x_shap.index].reset_index(drop = True)
```

`Customers with score < 300: 0`

The scaling procedure doesn't return values under 300 points. If such was the case we could trim our score to set a minimum of 300 or delete them if the number of such clients is small.

Let's plot the scores distributions by defaulters and non defaulters,


<img src="https://c-taimal.github.io/images/scores_dist.png" style="display: block; margin: auto;" />

While there is a small number of clients overlapping, overall, the score can distinguish between those charged-off and those who aren't. In the same manner, we see there are little clients that pass the threshold of 800 points, and without clients with scores under 300, this behavior is similar to that of the FICO.

Now, to check how good the score discriminates we compute the ROC-AUC. As a higher score indicates a lower probability of default, we must change its polarity to compute the metric and such that it coincides with the original probabilities:

```python
roc_auc_score(scores_df['default'], -scores_df['score'])
```

`ROC-AUC: 0.9672547105517366`

We obtain a ROC-AUC of almost 97%, such that the score correctly distinguish the nature of the clients. We verifiy this separation using the Kolmogorov-Smirnov test,

```python
kstest(scores_df.loc[scores_df['default'] == 0, 'score'],
       scores_df.loc[scores_df['default'] == 1, 'score'])
```

`KstestResult(statistic=np.float64(0.8738273190003616), pvalue=np.float64(0.0))`

With a separation magnitude of 87% and a formal verification by the test under a significance of 5%.

<img src="https://c-taimal.github.io/images/ks_scores.png" style="display: block; margin: auto;" />

Finally, using the points assignated to the 5 covariables we represent them in a two dimensional setting using the $$t$$-SNE and we use the labels by `loan_status` to color-separate our clients,

<img src="https://c-taimal.github.io/images/tsne_lgbm.png" style="display: block; margin: auto;" />

Overall this assignation allows us to perfectly separate between charged-off and non charged-off clients except for few cases that overlap.

And we have our final scorecard defined as follows,

<table>
    <thead>
        <tr>
            <th>Variable</th>
            <th>Category</th>
            <th>Points</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="5"><code>last_pymnt_amnt_binned</code></td>
            <td>(-inf, 99.62)</td>
            <td>-107</td>
        </tr>
        <tr>
            <td>[99.62, 1269.77)</td>
            <td>-72</td>
        </tr>
        <tr>
            <td>[1269.77, 2171.19)</td>
            <td>-14.00</td>
        </tr>
        <tr>
            <td>[2171.19, 5055.01)</td>
            <td>63</td>
        </tr>
        <tr>
            <td>[5055.01, inf)</td>
            <td>149</td>
        </tr>
        <tr>
            <td rowspan="5"><code>int_rate_binned</code></td>
            <td>(-inf, 7.88)</td>
            <td>36</td>
        </tr>
        <tr>
            <td>[7.88, 9.73)</td>
            <td>16</td>
        </tr>
        <tr>
            <td>[9.73, 13.18)</td>
            <td>-8</td>
        </tr>
        <tr>
            <td>[13.18, 17.78)</td>
            <td>-11</td>
        </tr>
        <tr>
            <td>[17.78, inf)</td>
            <td>-14</td>
        </tr>
        <tr>
            <td rowspan="7"><code>grade</code></td>
            <td>A</td>
            <td>31</td>
        </tr>
        <tr>
            <td>B</td>
            <td>9</td>
        </tr>
        <tr>
            <td>C</td>
            <td>-2</td>
        </tr>
        <tr>
            <td>D</td>
            <td>-16</td>
        </tr>
        <tr>
            <td>E</td>
            <td>-22</td>
        </tr>
        <tr>
            <td>F</td>
            <td>-31</td>
        </tr>
        <tr>
            <td>G</td>
            <td>-33</td>
        </tr>
        <tr>
            <td rowspan="2"><code>term</code></td>
            <td>36 months</td>
            <td>8</td>
        </tr>
        <tr>
            <td>60 months</td>
            <td>-27</td>
        </tr>
        <tr>
            <td rowspan="5"><code>acc_open_past_24mths_binned</code></td>
            <td>(-inf, 2.50)</td>
            <td>5</td>
        </tr>
        <tr>
            <td>[2.50, 4.50)</td>
            <td>0</td>
        </tr>
        <tr>
            <td>[4.50, 6.50)</td>
            <td>-2</td>
        </tr>
        <tr>
            <td>[6.50, 9.50)</td>
            <td>-3</td>
        </tr>
        <tr>
            <td>[9.50, inf)</td>
            <td>-5</td>
        </tr>
    </tbody>
</table>

Some points were rearranged to satisfy the monotonicy condition for the subgrupos of the features, i.e., larger values that delimit the bins assign larger points (exceptuando el caso de `grade`) and vice-versa. Beyond that, the model allowed, in general, to capture a distinction between defaulters and non defaulters, while the scaling process allows us to replicate a range similar to the one used by the FICO. 

As expected, one may ask themselves whether this method is superior or not to the classical ones. To answer this, we fit a Logistic Regression using the WoE an the model coefficients to create the scorecard.

### **Building the Scorecard the usual way**

For this process, and in contrast with our previous method, we use codify every category as WoE instead of binning. Again, we'll use the `optbinning` library,

```python
for var in numericas_final_ls:


    x = train_df[var]
    y = y_train
    optb = OptimalBinning(name = var, dtype="numerical", solver="cp", min_n_bins=2, max_n_bins=5)
    x_binned = optb.fit_transform(x, y, metric = 'woe')
    x_val_binned  = optb.transform(val_df[var], metric = 'woe')
    x_test_binned = optb.transform(test_df[var], metric = 'woe')

    train_df[var + '_woe'] = x_binned
    val_df[var + '_woe']   = x_val_binned
    test_df[var + '_woe']  = x_test_binned
```

the variables used here are the same as before: `last_pymnt_amnt`, `int_rate`,  `acc_open_past_24mths` under the WoE encoding along with `term` and `grade`.

```python
list_variables_model = ['term', 'grade'] + [var for var in train_df.columns if '_woe' in var]

list_variables_model = [var for var in list_variables_model if var != 'total_rec_late_fee_woe']
list_variables_model = [var for var in list_variables_model if var != 'recoveries_woe']
```

Besides this codification we must express `grade` and `term` as binary features,

```python
train_df = pd.get_dummies(train_df, columns = ['term', 'grade'], drop_first = True)
val_df = pd.get_dummies(val_df, columns = ['term', 'grade'], drop_first = True)
test_df = pd.get_dummies(test_df, columns = ['term', 'grade'], drop_first = True)
```

After this we can fit the model as we know it,

```python
from sklearn.linear_model import LogisticRegression

list_variables_model = [var for var in train_df.columns if ('term' in var) or ('grade_' in var) or ('_woe' in var)]

logit_classification = LogisticRegression()
logit_classification.fit(train_df[list_variables_model], y_train)
```
As with the LightGBM we tune the classification threshold using the validation set,

```python
y_probs_validation = logit_classification.predict_proba(val_df[list_variables_model])[:, 1]


recall_ls = [recall_score(y_val, np.where(y_probs_validation > t, 1, 0)) for t in thresholds]
f1_ls = [f1_score(y_val, np.where(y_probs_validation > t, 1, 0)) for t in thresholds]
pr_auc = [average_precision_score(y_val, np.where(y_probs_validation > t, 1, 0)) for t in thresholds]
```

```python
sns.lineplot(x = thresholds, y = recall_ls, label = 'Recall')
sns.scatterplot(x = thresholds, y = recall_ls)

sns.lineplot(x = thresholds, y = pr_auc, label = 'Precision AUC')
sns.scatterplot(x = thresholds, y = pr_auc)

sns.lineplot(x = thresholds, y = f1_ls, label = 'F-1')
sns.scatterplot(x = thresholds, y = f1_ls)
```
<img src="https://c-taimal.github.io/images/thresholds_logistic.png" style="display: block; margin: auto;" />

These metrics have a similar behavior to the LightGBM except they posses a larger range; we can easily choose a threshold of 0.5 where the metrics coincide before begin to degrade.

```python
y_test_probs = logit_classification.predict_proba(test_df[list_variables_model])[:, 1]
y_test_preds = np.where(y_test_probs > 0.5, 1, 0)

print(f'Recall Test: {recall_score(y_test, y_test_preds)} | PR-AUC Test: {average_precision_score(y_test, y_test_preds)} | Best F1: {f1_score(y_test, y_test_preds)}')
```

`Recall Test: 0.8353836040942811 | PR-AUC Test: 0.7814143782589272 | Best F1: 0.8636054751965828`

As expected, after binarizing `grade` and `term` the metrics will have a larger value in contrast with the LightGBM.

#### **From Logistic Regression to Scorecard**

Now we can define a function similar to the one used for the LightGBM to standardized the WoE and the model coefficients,

```python
def scoring_woe_values(
    score_0: float,
    odds_0: float,
    pdo: float,
    top_5_features: list,
    lr_model,
    test_df: pd.DataFrame,
):
  """Calculates scorecard points for WoE features based on logistic regression coefficients.

  Parameters:
  - score_0: Base score (e.g., 600)
  - odds_0: Base odds (e.g., odds of 50:1 meaning good/bad = 50)
  - pdo: Points to double odds (e.g., 20)
  - top_10_features: List of feature names
  - lr_model: Fitted LogisticRegression model containing .coef_ and .intercept_
  - X_train_woe: Training dataframe containing the WoE transformed columns
  """

  b = pdo / np.log(2)  # Note: standard scorecard sign convention for b
  score_base = score_0 - b * np.log(odds_0)

  scorecard_tables = {}
  variables_score = test_df[top_5_features].copy()

  # Extract coefficients from the logistic regression model
  # Assuming lr_model.coef_ is a 1D array or 2D array matching top_10_features order
  feature_coefs = dict(
      zip(
          lr_model.feature_names_in_
          if hasattr(lr_model, 'feature_names_in_')
          else top_5_features,
          lr_model.coef_[0] if lr_model.coef_.ndim > 1 else lr_model.coef_,
      )
  )

  for col in top_5_features:
    coef = feature_coefs.get(
        col, 1.0
    )  

    # Since WoE is already in the dataset, we map unique WoE values back to their categories or bin values
    # If X_train_woe[col] already holds the WoE numerical value:
    # We group by the WoE value itself to compute points per WoE bucket

    df_woe = pd.DataFrame({'category': test_df[col], 'woe': test_df[col]})

    category_summary = (df_woe.groupby('category', observed=False)['woe'].mean().reset_index())

    # Scorecard point formula for WoE: Points = -b * (WoE * beta + Intercept / n_features)
    # Distributing the intercept evenly across features is standard practice in credit scorecards

    intercept_share = lr_model.intercept_[0] / len(top_5_features)
    category_summary['points'] = np.round(-b * (category_summary['woe'] * coef + intercept_share))

    scorecard_tables[col] = category_summary

  # Map points to the dataset variables
  for col in top_5_features:
    table = scorecard_tables[col]
    points_map = table.set_index('category')['points'].to_dict()
    variables_score[col] = variables_score[col].map(points_map)

  # Format final summary tables for export
  final_scorecards = []
  for col, table in scorecard_tables.items():
    table = table.copy()
    table.insert(0, 'feature', col)
    final_scorecards.append(table)

  return score_base, variables_score, final_scorecards
```

The same five variables used for the LightGBM are here used once again and the  $$Score_0$$, $$Odds_0$$ and $$\text{PdO}$$ are the same as for the SHAP values scorecard,

```python
top_5_features_2 = ['last_pymnt_amnt_woe', 'int_rate_woe', 'grade_B', 'grade_C',
                    'grade_D', 'grade_E', 'grade_F', 'grade_G', 'term_ 60 months', 'acc_open_past_24mths_woe']

score_base_lr, scores_df_lr, scorecard_lr = scoring_woe_values(600,50, 20, top_5_features_2, logit_classification, test_df.loc[x_shap.index])

scores_df_lr['score'] = score_base_lr + scores_df_lr.sum(axis = 1)
scores_df_lr = scores_df_lr.reset_index(drop = True)
scores_df_lr['default'] = y_test.loc[x_shap.index].reset_index(drop = True)
```

Let's compute the same metrics as for the LightGBM score the ROC-AUC and the KS $$p$$-value and $$D$$-statistic:

```python
roc_auc_score(scores_df_lr['default'], -scores_df_lr['score'])
```

`ROC-AUC: 0.9655030732538893`

We obtain a ROC-AUC value that is barely lower than one for the SHAP scorecard. After implementing the Kolmogorov-Smirnov test we find, again, a good separation through the $D$ statistic as with the SHAP scorecard case,

```python
kstest(scores_df_lr.loc[scores_df_lr['default'] == 0, 'score'],
       scores_df_lr.loc[scores_df_lr['default'] == 1, 'score'])
```

`KstestResult(statistic=np.float64(0.8667353300072896), pvalue=np.float64(0.0))`

```python
sns.ecdfplot(
        data=scores_df_lr,
        x='score',
        hue="default",
        linewidth=1.5
)
plt.ylim((0, 1.1))
```

<img src="https://c-taimal.github.io/images/ks_scores_logistic.png" style="display: block; margin: auto;" />

And to conclude the separability analysis, we make a bidimensional representation for the points per variable using the $t$-SNE,

<img src="https://c-taimal.github.io/images/tsne_logistic.png" style="display: block; margin: auto;" />

Again, there is a similar pattern for the defaulter clients in the left with a small quantity of clients overlapping with those who aren't defaulters.

Under the SHAP values there seems to be non significant differences to using a logistic regression. Next, let's discuss these findings and to conclude whether one or the other is better.

## **Conclusions**

After fitting a Logistic Regression under similar conditions and computing the values to obtain a scorecard, we obtain similar results to the ones we can obtain after using a LightGBM, nevertheless defining one method or another as the best is not easy as it depends on the business needs.


First of all, using the LightGBM allows us to maintain the qualitative variables categorical structure. Even if we could encode our numerical variables through the WoE, for the logistic regression we had to create dummies for `grade` and `term` adding more dimension to the data matrix and with it, a possible overfitting.


On the other hand the WoE encoding is not strictly bounded and depends on the original variable (TIBCO, 2026)[^2]. For example, the WoE for `last_pymnt_amnt_woe` goes from -24 to around 200, while for some grades, e.g., `grade_f` it barely shows dispersion ranging from -8 to 8. So an additional checking on the variables range will be necessary to ensure a "normal" behavior for them as extreme values can sometimes interfere with the results of the logistic regression (Lukman et al., 2024)[^3].

In the case of using a LightGBM classifier we have a more robustness in regards of possible outliers and extreme values. Also we can proceed with large correlated features and finally, both models can be easily scalabe to large datasets so the choice should not rely only on optimizing a metric.

As we spoke earlier, some regulations define rules to implement risk models, so when it comes to interpretability and audits a logistic regression is preferred. But if we can use any model and we want robustness and modifying our data the less, a tree-based approach like LightGBM can be used.

Nevertheless, the here presented approach can be used as a bridge between both type of implementations while making it possible to address the lack of interpretability that some "black box"  models are often criticized for.

# **Bibliography**

[^1]: Navas-Palencia, G. (2022). Optimal binning: mathematical programming formulation. Arxiv preprint: https://arxiv.org/abs/2001.08025

[^2]: TIBCO . (2026). TIBCO Statistica® 14.1.0 Product Documentation. https://docs.tibco.com/pub/stat/14.1.0/doc/html/UserGuide/_shared/weight-of-evidence-woe-overview.htm

[^3]: Lukman, A. F., Mohammed, S., Olaluwoye, O. & Farghali, R.A. (2024). Handling Multicollinearity and Outliers in Logistic Regression Using the Robust Kibria–Lukman Estimator. *Axioms*. 14 (1). 


import numpy as np
import pandas as pd
from scipy.stats import ranksums, chisquare, kruskal, alexandergovern, kendalltau, f_oneway, shapiro, kstest, anderson, normaltest, boxcox, yeojohnson, bartlett, levene, fligner, obrientransform, pearsonr, spearmanr, pointbiserialr, ttest_ind, mannwhitneyu, wilcoxon,ttest_rel
from typing import Optional, Union, List
from statsmodels.stats.multitest import multipletests
from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI, Path, Query, APIRouter
import sklearn
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, SGDClassifier, HuberRegressor,Lars
from sklearn.svm import SVR, LinearSVR, LinearSVC
from pingouin import ancova
import statsmodels.api as sm
import statsmodels.formula.api as smf

router = APIRouter()
data = pd.read_csv('example_data/mescobrad_dataset.csv')
@router.get("/return_columns")
async def name_columns():
    columns = data.columns
    return{'columns': list(columns)}


@router.get("/normality_tests", tags=['hypothesis_testing'])
async def normal_tests(column: str,
                       name_test: str | None = Query("Shapiro-Wilk",
                                                   regex="^(Shapiro-Wilk)$|^(Kolmogorov-Smirnov)$|^(Anderson-Darling)$|^(D’Agostino’s K\^2)$")) -> dict:
    if name_test == 'Shapiro-Wilk':
        shapiro_test = shapiro(data[str(column)])
        if shapiro_test.pvalue > 0.05:
            return{'statistic': shapiro_test.statistic, 'p_value': shapiro_test.pvalue, 'Description': 'Sample looks Gaussian (fail to reject H0)'}
        else:
            return{'statistic': shapiro_test.statistic, 'p_value': shapiro_test.pvalue, 'Description':'Sample does not look Gaussian (reject H0)'}
    elif name_test == 'Kolmogorov-Smirnov':
        ks_test = kstest(data[str(column)], 'norm')
        if ks_test.pvalue > 0.05:
            return{'statistic': ks_test.statistic, 'p_value': ks_test.pvalue, 'Description':'Sample looks Gaussian (fail to reject H0)'}
        else:
            return{'statistic': ks_test.statistic, 'p_value': ks_test.pvalue, 'Description':'Sample does not look Gaussian (reject H0)'}
    elif name_test == 'Anderson-Darling':
        anderson_test = anderson(data[str(column)])
        list_anderson = []
        for i in range(len(anderson_test.critical_values)):
            sl, cv = anderson_test.significance_level[i], anderson_test.critical_values[i]
            if anderson_test.statistic < anderson_test.critical_values[i]:
                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
                list_anderson.append('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
            else:
                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
                list_anderson.append('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
        return{'statistic':anderson_test.statistic, 'critical_values': list(anderson_test.critical_values), 'significance_level': list(anderson_test.significance_level), 'description': list_anderson}
    elif name_test == 'D’Agostino’s K^2':
        stat, p = normaltest(data[str(column)])
        if p > 0.05:
            return{'statistic': stat, 'p_value': p, 'Description':'Sample looks Gaussian (fail to reject H0)'}
        else:
            return{'statistic': stat, 'p_value': p, 'Description':'Sample does not look Gaussian (reject H0)'}

@router.get("/transform_data")
async def transform_data(column:str,
                         name_transform:str | None = Query("Box-Cox",
                                                           regex="^(Box-Cox)$|^(Yeo-Johnson)$"),
                         lmbd: Optional[float] = None,
                         alpha: Optional[float] = None) -> dict:
    if name_transform == 'Box-Cox':
        if lmbd == None:
            if alpha == None:
                boxcox_array, maxlog = boxcox(np.array(data[str(column)]))
                return {'Box-Cox power transformed array': list(boxcox_array), 'lambda that maximizes the log-likelihood function': maxlog}
            else:
                boxcox_array, maxlog, z = boxcox(np.array(data[str(column)]), alpha = alpha)
                return {'Box-Cox power transformed array': list(boxcox_array), 'lambda that maximizes the log-likelihood function': maxlog, 'minimum confidence limit': z[0], 'maximum confidence limit': z[1]}
        else:
            if alpha == None:
                y = boxcox(np.array(data[str(column)]), lmbda = lmbd)
                return {'Box-Cox power transformed array': list(y)}
            else:
                y = boxcox(np.array(data[str(column)]), lmbda=lmbd, alpha = alpha)
                return {'Box-Cox power transformed array': list(y)}
    elif name_transform == 'Yeo-Johnson':
        if lmbd == None:
            yeojohnson_array, maxlog = yeojohnson(np.array(data[str(column)]))
            return {'Yeo-Johnson power transformed array': list(yeojohnson_array), 'lambda that maximizes the log-likelihood function': maxlog}
        else:
            yeojohnson_array = yeojohnson(np.array(data[str(column)]), lmbda=lmbd)
            return {'Yeo-Johnson power transformed array': list(yeojohnson_array)}

@router.get("/compute_pearson_correlation")
async def pearson_correlation(column_1: str, column_2: str):
    pearsonr_test = pearsonr(data[str(column_1)], data[str(column_2)])
    return {'Pearson’s correlation coefficient':pearsonr_test[0], 'p-value': pearsonr_test[1]}

@router.get("/compute_spearman_correlation")
async def spearman_correlation(column_1: str, column_2: str):
    spearman_test = spearmanr(data[str(column_1)], data[str(column_2)])
    return {'Spearman correlation coefficient': spearman_test[0], 'p-value': spearman_test[1]}

@router.get("/compute_kendalltau_correlation")
async def kendalltau_correlation(column_1: str,
                                 column_2: str,
                                 alternative: Optional[str] | None = Query("two-sided",
                                                                           regex="^(two-sided)$|^(less)$|^(greater)$"),
                                 variant: Optional[str] | None = Query("b",
                                                                       regex="^(b)$|^(c)$"),
                                 method: Optional[str] | None = Query("auto",
                                                                      regex="^(auto)$|^(asymptotic)$|^(exact)$")):
    kendalltau_test = kendalltau(data[str(column_1)], data[str(column_2)], alternative=alternative, variant=variant, method=method)
    return {'kendalltau correlation coefficient': kendalltau_test[0], 'p-value': kendalltau_test[1]}

@router.get("/compute_point_biserial_correlation")
async def point_biserial_correlation(column_1: str, column_2: str):
    unique_values = np.unique(data[str(column_1)])
    if len(unique_values) == 2:
        pointbiserialr_test = pointbiserialr(data[str(column_1)], data[str(column_2)])
    else:
        pointbiserialr_test = pointbiserialr(data[str(column_2)], data[str(column_1)])
    return {'correlation':pointbiserialr_test[0], 'p-value': pointbiserialr_test[1]}

#
@router.get("/check_homoscedasticity")
async def check_homoskedasticity(column_1: str,
                           column_2: str,
                           name_of_test: str | None = Query("Levene",
                                                           regex="^(Levene)$|^(Bartlett)$|^(Fligner-Killeen)$"),
                           center: Optional[str] | None = Query("median",
                                                      regex="^(trimmed)$|^(median)$|^(mean)$")):
    if name_of_test == "Bartlett":
        statistic, p_value = bartlett(data[str(column_1)], data[str(column_2)])
    elif name_of_test == "Fligner-Killeen":
        statistic, p_value = fligner(data[str(column_1)], data[str(column_2)], center=center)
    else:
        statistic, p_value = levene(data[str(column_1)], data[str(column_2)], center = center)
    return {'statistic': statistic, 'p-value': p_value}

@router.get("/transformed_data_for_use_in_an_ANOVA")
async def transform_data_anova(column_1: str, column_2: str):
    tx, ty = obrientransform(data[str(column_1)], data[str(column_2)])
    return {'transformed_1': list(tx), 'transformed_2': list(ty)}


@router.get("/statistical_tests")
async def statistical_tests(column_1: str,
                            column_2: str,
                            correction: bool,
                            statistical_test: str | None = Query("Independent t-test",
                                                                 regex="^(Independent t-test)$|^(Welch t-test)$|^(Mann-Whitney U rank test)$|^(t-test on TWO RELATED samples of scores)$|^(Wilcoxon signed-rank test)$|^(Alexander Govern test)$|^(Kruskal-Wallis H-test)$|^(one-way ANOVA)$|^(Wilcoxon rank-sum statistic)$|^(one-way chi-square test)$"),
                            alternative: Optional[str] | None = Query("two-sided",
                                                                      regex="^(two-sided)$|^(less)$|^(greater)$"),
                            method: Optional[str] | None = Query("auto",
                                                                 regex="^(auto)$|^(asymptotic)$|^(exact)$"),
                            mode: Optional[str] | None = Query("auto",
                                                                 regex="^(auto)$|^(approx)$|^(exact)$"),
                            zero_method: Optional[str] | None = Query("pratt",
                                                                 regex="^(pratt)$|^(wilcox)$|^(zsplit)$")):

    if statistical_test == "Welch t-test":
        statistic, p_value = ttest_ind(data[str(column_1)], data[str(column_2)], equal_var=False, alternative=alternative)
    elif statistical_test == "Independent t-test":
        statistic, p_value = ttest_ind(data[str(column_1)], data[str(column_2)], alternative=alternative)
    elif statistical_test == "t-test on TWO RELATED samples of scores":
        if np.shape(data[str(column_1)])[0] != np.shape(data[str(column_2)])[0]:
            return {'error': 'Unequal length arrays'}
        statistic, p_value = ttest_rel(data[str(column_1)], data[str(column_2)], alternative=alternative)
    elif statistical_test == "Mann-Whitney U rank test":
        statistic, p_value = mannwhitneyu(data[str(column_1)], data[str(column_2)], alternative=alternative, method=method)
    elif statistical_test == "Wilcoxon signed-rank test":
        if np.shape(data[str(column_1)])[0] != np.shape(data[str(column_2)])[0]:
            return {'error': 'Unequal length arrays'}
        statistic, p_value = wilcoxon(data[str(column_1)], data[str(column_2)], alternative=alternative, correction=correction, zero_method=zero_method, mode=mode)
    elif statistical_test == "Alexander Govern test":
        z = alexandergovern(data[str(column_1)], data[str(column_2)])
        return {'mean_positive': np.mean(data[str(column_1)]), 'standard_deviation_positive': np.std(data[str(column_1)]),
                'mean_negative': np.mean(data[str(column_2)]), 'standard_deviation_negative': np.std(data[str(column_2)]),
                'statistic, p_value': z}
    elif statistical_test == "Kruskal-Wallis H-test":
        statistic, p_value = kruskal(data[str(column_1)], data[str(column_2)])
    elif statistical_test == "one-way ANOVA":
        statistic, p_value = f_oneway(data[str(column_1)], data[str(column_2)])
    elif statistical_test == "Wilcoxon rank-sum statistic":
        statistic, p_value = ranksums(data[str(column_1)], data[str(column_2)])
    elif statistical_test == "one-way chi-square test":
        statistic, p_value = chisquare(data[str(column_1)], data[str(column_2)])
    return {'mean_positive': np.mean(data[str(column_1)]), 'standard_deviation_positive': np.std(data[str(column_1)]),
            'mean_negative': np.mean(data[str(column_2)]), 'standard_deviation_negative': np.std(data[str(column_2)]),
            'statistic': statistic, 'p-value': p_value}


@router.get("/multiple_comparisons")
async def p_value_correction(alpha: float,
                             p_value: list[float] = Query([]),
                             method: str | None = Query("Bonferroni",
                                                        regex="^(Bonferroni)$|^(sidak)$|^(holm-sidak)$|^(holm)$|^(simes-hochberg)$|^(benjamini-hochberg)$|^(benjamini-yekutieli)$|^(fdr_tsbh)$|^(fdr_tsbky)$")):
    if method == 'Bonferroni':
        z = multipletests(pvals=p_value, alpha=alpha, method='bonferroni')
        y = [str(x) for x in z[0]]
        return {'true for hypothesis that can be rejected for given alpha': list(y), 'corrected_p_values': list(z[1])}
    elif method == 'sidak':
        z = multipletests(pvals=p_value, alpha=alpha, method='sidak')
        y = [str(x) for x in z[0]]
        return {'true for hypothesis that can be rejected for given alpha': list(y), 'corrected_p_values': list(z[1])}
    elif method == 'benjamini-hochberg':
        z = multipletests(pvals=p_value, alpha=alpha, method='fdr_bh')
        y = [str(x) for x in z[0]]
        return {'true for hypothesis that can be rejected for given alpha': list(y), 'corrected_p_values': list(z[1])}
    elif method == 'benjamini-yekutieli':
        z = multipletests(pvals=p_value, alpha=alpha, method='fdr_by')
        y = [str(x) for x in z[0]]
        return {'true for hypothesis that can be rejected for given alpha': list(y), 'corrected_p_values': list(z[1])}
    else:
        z = multipletests(pvals=p_value, alpha=alpha, method= method)
        y = [str(x) for x in z[0]]
        return {'true for hypothesis that can be rejected for given alpha': list(y), 'corrected_p_values': list(z[1])}

@router.get("/LDA")
async def LDA(dependent_variable: str,
              solver: str | None = Query("svd",
                                         regex="^(svd)$|^(lsqr)$|^(eigen)$"),
              shrinkage_1: str | None = Query(None,
                                              regex="^(None)$|^(auto)$"),
              shrinkage_2: float | None = Query(default=None, gt=0, lt=0),
              shrinkage_3 : float | None = Query(default=None),
              independent_variables: list[str] | None = Query(default=None)):

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    if solver == 'lsqr' or solver == 'eigen':
        if shrinkage_3==None:
            clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage_3)
        elif shrinkage_1!=None:
            clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage_1)
        else:
            clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage_2)
    else:
        clf = LinearDiscriminantAnalysis(solver=solver)

    clf.fit(X, Y)
    coeffs = np.squeeze(clf.coef_)
    inter = clf.intercept_
    return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist()}

@router.get("/SVC")
async def SVC(dependent_variable: str,
              degree: int | None = Query(default=3),
              max_iter: int | None = Query(default=-1),
              epsilon: float | None = Query(default=0.1),
              C: float | None = Query(default=1,gt=0),
              coef0: float | None = Query(default=0),
              gamma: str | None = Query("scale",
                                        regex="^(scale)$|^(auto)$"),
              kernel: str | None = Query("rbf",
                                         regex="^(rbf)$|^(linear)$|^(poly)$|^(sigmoid)$|^(precomputed)$"),
              independent_variables: list[str] | None = Query(default=None)):

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    if kernel == 'poly':
        clf = SVC(degree=degree, gamma=gamma, coef0=coef0, C=C, epsilon=epsilon, max_iter=max_iter)
    elif kernel == 'rbf' or kernel == 'sigmoid':
        if kernel == 'sigmoid':
            clf = SVC(gamma=gamma, coef0=coef0, C=C, epsilon=epsilon, max_iter=max_iter)
        else:
            clf = SVC(gamma=gamma, C=C, epsilon=epsilon, max_iter=max_iter)

    clf.fit(X, Y)
    if kernel == 'linear':
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist()}
    else:
        coeffs = np.squeeze(clf.dual_coef_)
        inter = clf.intercept_
        return {'Dual coefficients': coeffs.tolist(), 'intercept': inter.tolist()}


@router.get("/principal_component_analysis")
async def principal_component_analysis(n_components_1: int | None = Query(default=None),
                                       n_components_2: float | None = Query(default=None, gt=0, lt=1),
                                       independent_variables: list[str] | None = Query(default=None)):
    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    list_1 = []
    list_1.append(int(np.shape(X)[0]))
    list_1.append(int(np.shape(X)[1]))
    dim = min(list_1)

    if n_components_2 == None:
        if n_components_1 > dim:
            return {'Error: n_components must be between 0 and min(n_samples, n_features)=': dim}
        pca = PCA(n_components=n_components_1)
        pca.fit(X)
    else:
        pca = PCA(n_components=n_components_2)
        pca.fit(X)

    return {'Percentage of variance explained by each of the selected components': pca.explained_variance_ratio_.tolist(),
            'The singular values corresponding to each of the selected components. ': pca.singular_values_.tolist(),
            'Principal axes in feature space, representing the directions of maximum variance in the data.' : pca.components_.tolist()}

@router.get("/kmeans_clustering")
async def kmeans_clustering(n_clusters: int,
                            independent_variables: list[str] | None = Query(default=None)):
    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    return {'Coordinates of cluster centers': kmeans.cluster_centers_.tolist(),
            'Labels of each point ': kmeans.labels_.tolist(),
            'Sum of squared distances of samples to their closest cluster center' : kmeans.inertia_}

@router.get("/linear_regressor")
async def linear_regression(dependent_variable: str,
                            independent_variables: list[str] | None = Query(default=None)):
    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    clf = LinearRegression()

    clf.fit(X, Y)
    coeffs = np.squeeze(clf.coef_)
    inter = clf.intercept_
    return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist()}

@router.get("/elastic_net")
async def elastic_net(dependent_variable: str,
                      alpha: float | None = Query(default=1.0),
                      l1_ratio: float | None = Query(default=0.5, ge=0, le=1),
                      max_iter: int | None = Query(default=1000),
                      independent_variables: list[str] | None = Query(default=None)):



    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)

    clf.fit(X, Y)
    coeffs = np.squeeze(clf.coef_)
    inter = clf.intercept_
    return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist()}

@router.get("/lasso_regression")
async def lasso(dependent_variable: str,
                alpha: float | None = Query(default=1.0, gt=0),
                max_iter: int | None = Query(default=1000),
                independent_variables: list[str] | None = Query(default=None)):

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    clf = Lasso(alpha=alpha, max_iter=max_iter)

    clf.fit(X, Y)
    coeffs = np.squeeze(clf.coef_)
    inter = clf.intercept_
    return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist()}

@router.get("/ridge_regression")
async def ridge(dependent_variable: str,
                alpha: float | None = Query(default=1.0, gt=0),
                max_iter: int | None = Query(default=None),
                solver: str | None = Query("auto",
                                           regex="^(auto)$|^(svd)$|^(cholesky)$|^(sparse_cg)$|^(lsqr)$|^(sag)$|^(lbfgs)$"),
                independent_variables: list[str] | None = Query(default=None)):

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    if solver!='lbfgs':
        clf = Ridge(alpha=alpha, max_iter=max_iter, solver=solver)
    else:
        clf = Ridge(alpha=alpha, max_iter=max_iter, solver=solver, positive=True)

    clf.fit(X, Y)
    coeffs = np.squeeze(clf.coef_)
    inter = clf.intercept_
    return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist()}

@router.get("/sgd_regression")
async def sgd_regressor(dependent_variable: str,
                        alpha: float | None = Query(default=0.0001),
                        max_iter: int | None = Query(default=1000),
                        epsilon: float | None = Query(default=0.1),
                        eta0: float | None = Query(default=0.01),
                        l1_ratio: float | None = Query(default=0.15, ge=0, le=1),
                        loss: str | None = Query("squared_error",
                                                   regex="^(squared_error)$|^(huber)$|^(epsilon_insensitive)$|^(squared_epsilon_insensitive)$"),
                        learning_rate: str | None = Query("invscaling",
                                                   regex="^(invscaling)$|^(constant)$|^(optimal)$|^(adaptive)$"),
                        penalty: str | None = Query("l2",
                                                 regex="^(l2)$|^(l1)$|^(elasticnet)$"),
                        independent_variables: list[str] | None = Query(default=None)):

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    if loss == 'huber' or loss == 'epsilon_insensitive' or loss == 'squared_epsilon_insensitive':
        if learning_rate == 'constant' or learning_rate == 'invscaling' or learning_rate == 'adaptive':
            clf = SGDRegressor(alpha=alpha, max_iter=max_iter, epsilon=epsilon, eta0=eta0, penalty=penalty, l1_ratio=l1_ratio, learning_rate=learning_rate)
        else:
            clf = SGDRegressor(alpha=alpha, max_iter=max_iter, epsilon=epsilon, penalty=penalty, l1_ratio=l1_ratio, learning_rate=learning_rate)
    else:
        clf = SGDRegressor(alpha=alpha, max_iter=max_iter, eta0=eta0, penalty=penalty, l1_ratio=l1_ratio, learning_rate=learning_rate)

    clf.fit(X, Y)
    coeffs = np.squeeze(clf.coef_)
    inter = clf.intercept_
    return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist()}

@router.get("/huber_regression")
async def huber_regressor(dependent_variable: str,
                          max_iter: int | None = Query(default=1000),
                          epsilon: float | None = Query(default=1.5, gt=1),
                          alpha: float | None = Query(default=0.0001,ge=0),
                          independent_variables: list[str] | None = Query(default=None)):

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    clf = HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=max_iter)

    clf.fit(X, Y)
    coeffs = np.squeeze(clf.coef_)
    inter = clf.intercept_
    outliers = clf.outliers_
    return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'outliers':outliers.tolist()}

@router.get("/svr_regression")
async def svr_regressor(dependent_variable: str,
                        degree: int | None = Query(default=3),
                        max_iter: int | None = Query(default=-1),
                        epsilon: float | None = Query(default=0.1),
                        C: float | None = Query(default=1,gt=0),
                        coef0: float | None = Query(default=0),
                        gamma: str | None = Query("scale",
                                                   regex="^(scale)$|^(auto)$"),
                        kernel: str | None = Query("rbf",
                                                 regex="^(rbf)$|^(linear)$|^(poly)$|^(sigmoid)$|^(precomputed)$"),

                        independent_variables: list[str] | None = Query(default=None)):

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    if kernel == 'poly':
        clf = SVR(degree=degree, gamma=gamma, coef0=coef0, C=C, epsilon=epsilon, max_iter=max_iter)
    elif kernel == 'rbf' or kernel == 'sigmoid':
        if kernel == 'sigmoid':
            clf = SVR(gamma=gamma, coef0=coef0, C=C, epsilon=epsilon, max_iter=max_iter)
        else:
            clf = SVR(gamma=gamma, C=C, epsilon=epsilon, max_iter=max_iter)

    clf.fit(X, Y)
    if kernel == 'linear':
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist()}
    else:
        coeffs = np.squeeze(clf.dual_coef_)
        inter = clf.intercept_
        return {'Coefficients of the support vector in the decision function.': coeffs.tolist(), 'intercept': inter.tolist()}

@router.get("/linearsvr_regression")
async def linear_svr_regressor(dependent_variable: str,
                               max_iter: int | None = Query(default=1000),
                               epsilon: float | None = Query(default=0),
                               C: float | None = Query(default=1,gt=0),
                               loss: str | None = Query("epsilon_insensitive",
                                                         regex="^(epsilon_insensitive)$|^(squared_epsilon_insensitive)$"),
                               independent_variables: list[str] | None = Query(default=None)):

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    clf = LinearSVR(loss=loss, C=C, epsilon=epsilon, max_iter=max_iter)

    clf.fit(X, Y)
    coeffs = np.squeeze(clf.coef_)
    inter = clf.intercept_
    return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist()}

@router.get("/linearsvc_regression")
async def linear_svc_regressor(dependent_variable: str,
                               max_iter: int | None = Query(default=1000),
                               C: float | None = Query(default=1,gt=0),
                               loss: str | None = Query("hinge",
                                                         regex="^(hinge)$|^(squared_hinge)$"),
                               penalty: str | None = Query("l2",
                                                         regex="^(l1)$|^(l2)$"),
                               independent_variables: list[str] | None = Query(default=None)):

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    if loss == 'hinge' and penalty == 'l1':
        return {'This combination is not supported.'}
    else:
        clf = LinearSVC(loss=loss, C=C, penalty=penalty, max_iter=max_iter)

    clf.fit(X, Y)
    coeffs = np.squeeze(clf.coef_)
    inter = clf.intercept_
    return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist()}

@router.get("/ancova")
async def ancova_2(dv: str,
                   between: str,
                   covar: list[str] | None = Query(default=None),
                   effsize: str | None = Query("np2",
                                               regex="^(np2)$|^(n2)$")):

    df_data = pd.read_csv('example_data/mescobrad_dataset.csv')

    df = ancova(data=df_data, dv=dv, covar=covar, between=between, effsize=effsize)

    return {'ANCOVA':df.to_json(orient="split")}

@router.get("/linear_mixed_effects_model")
async def linear_mixed_effects_model(dependent: str,
                                     groups: str,
                                     independent: list[str] | None = Query(default=None),
                                     use_sqrt: bool | None = Query(default=True)):

    data = pd.read_csv('example_data/mescobrad_dataset.csv')

    z = dependent + "~"
    for i in range(len(independent)):
        z = z + "+" + independent[i]

    md = smf.mixedlm(z, data, groups=data[groups], use_sqrt=use_sqrt)

    mdf = md.fit()

    df = mdf.summary()

    print(df)

    return {df}

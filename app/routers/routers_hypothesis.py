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

from app.pydantic_models import ModelMultipleComparisons

router = APIRouter()
data = pd.read_csv('example_data/sample_questionnaire.csv')

@router.get("/return_columns")
async def name_columns():
    columns = data.columns
    return{'columns': list(columns)}


@router.get("/normality_tests", tags=['hypothesis_testing'])
async def normal_tests(column: str,
                       nan_policy: Optional[str] | None = Query("propagate",
                                                                regex="^(propagate)$|^(raise)$|^(omit)$"),
                       axis: Optional[int] = 0,
                       alternative: Optional[str] | None = Query("two-sided",
                                                                 regex="^(two-sided)$|^(less)$|^(greater)$"),
                       name_test: str | None = Query("Shapiro-Wilk",
                                                   regex="^(Shapiro-Wilk)$|^(Kolmogorov-Smirnov)$|^(Anderson-Darling)$|^(D’Agostino’s K\^2)$")) -> dict:
    if name_test == 'Shapiro-Wilk':
        shapiro_test = shapiro(data[str(column)])
        if shapiro_test.pvalue > 0.05:
            return{'statistic': shapiro_test.statistic, 'p_value': shapiro_test.pvalue, 'Description': 'Sample looks Gaussian (fail to reject H0)', 'data': data[str(column)].tolist()}
        else:
            return{'statistic': shapiro_test.statistic, 'p_value': shapiro_test.pvalue, 'Description':'Sample does not look Gaussian (reject H0)', 'data': data[str(column)].tolist()}
    elif name_test == 'Kolmogorov-Smirnov':
        ks_test = kstest(data[str(column)], 'norm', alternative=alternative)
        if ks_test.pvalue > 0.05:
            return{'statistic': ks_test.statistic, 'p_value': ks_test.pvalue, 'Description':'Sample looks Gaussian (fail to reject H0)', 'data': data[str(column)].tolist()}
        else:
            return{'statistic': ks_test.statistic, 'p_value': ks_test.pvalue, 'Description':'Sample does not look Gaussian (reject H0)', 'data': data[str(column)].tolist()}
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
        return{'statistic':anderson_test.statistic, 'critical_values': list(anderson_test.critical_values), 'significance_level': list(anderson_test.significance_level), 'Description': list_anderson, 'data': data[str(column)].tolist()}
    elif name_test == 'D’Agostino’s K^2':
        stat, p = normaltest(data[str(column)], nan_policy=nan_policy)
        if p > 0.05:
            return{'statistic': stat, 'p_value': p, 'Description':'Sample looks Gaussian (fail to reject H0)', 'data': data[str(column)].tolist()}
        else:
            return{'statistic': stat, 'p_value': p, 'Description':'Sample does not look Gaussian (reject H0)', 'data': data[str(column)].tolist()}

@router.get("/transform_data", tags=['hypothesis_testing'])
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

@router.get("/compute_pearson_correlation", tags=['hypothesis_testing'])
async def pearson_correlation(column_1: str, column_2: str):
    pearsonr_test = pearsonr(data[str(column_1)], data[str(column_2)])
    return {'Pearson’s correlation coefficient':pearsonr_test[0], 'p-value': pearsonr_test[1]}

@router.get("/compute_spearman_correlation", tags=['hypothesis_testing'])
async def spearman_correlation(column_1: str, column_2: str):
    spearman_test = spearmanr(data[str(column_1)], data[str(column_2)])
    return {'Spearman correlation coefficient': spearman_test[0], 'p-value': spearman_test[1]}

@router.get("/compute_kendalltau_correlation", tags=['hypothesis_testing'])
async def kendalltau_correlation(column_1: str,
                                 column_2: str,
                                 nan_policy: Optional[str] | None = Query("propagate",
                                                                           regex="^(propagate)$|^(raise)$|^(omit)$"),
                                 alternative: Optional[str] | None = Query("two-sided",
                                                                           regex="^(two-sided)$|^(less)$|^(greater)$"),
                                 variant: Optional[str] | None = Query("b",
                                                                       regex="^(b)$|^(c)$"),
                                 method: Optional[str] | None = Query("auto",
                                                                      regex="^(auto)$|^(asymptotic)$|^(exact)$")):
    kendalltau_test = kendalltau(data[str(column_1)], data[str(column_2)], nan_policy=nan_policy, alternative=alternative, variant=variant, method=method)
    return {'kendalltau correlation coefficient': kendalltau_test[0], 'p-value': kendalltau_test[1]}

@router.get("/compute_point_biserial_correlation", tags=['hypothesis_testing'])
async def point_biserial_correlation(column_1: str, column_2: str):
    unique_values = np.unique(data[str(column_1)])
    if len(unique_values) == 2:
        pointbiserialr_test = pointbiserialr(data[str(column_1)], data[str(column_2)])
    else:
        pointbiserialr_test = pointbiserialr(data[str(column_2)], data[str(column_1)])
    return {'correlation':pointbiserialr_test[0], 'p-value': pointbiserialr_test[1]}

#
@router.get("/check_homoscedasticity", tags=['hypothesis_testing'])
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

@router.get("/transformed_data_for_use_in_an_ANOVA", tags=['hypothesis_testing'])
async def transform_data_anova(column_1: str, column_2: str):
    tx, ty = obrientransform(data[str(column_1)], data[str(column_2)])
    return {'transformed_1': list(tx), 'transformed_2': list(ty)}


@router.get("/statistical_tests", tags=['hypothesis_testing'])
async def statistical_tests(column_1: str,
                            column_2: str,
                            correction: bool = True,
                            nan_policy: Optional[str] | None = Query("propagate",
                                                                     regex="^(propagate)$|^(raise)$|^(omit)$"),
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
        statistic, p_value = ttest_ind(data[str(column_1)], data[str(column_2)], nan_policy=nan_policy, equal_var=False, alternative=alternative)
    elif statistical_test == "Independent t-test":
        statistic, p_value = ttest_ind(data[str(column_1)], data[str(column_2)], nan_policy=nan_policy, alternative=alternative)
    elif statistical_test == "t-test on TWO RELATED samples of scores":
        if np.shape(data[str(column_1)])[0] != np.shape(data[str(column_2)])[0]:
            return {'error': 'Unequal length arrays'}
        statistic, p_value = ttest_rel(data[str(column_1)], data[str(column_2)], nan_policy=nan_policy, alternative=alternative)
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
        statistic, p_value = kruskal(data[str(column_1)], data[str(column_2)], nan_policy=nan_policy)
    elif statistical_test == "one-way ANOVA":
        statistic, p_value = f_oneway(data[str(column_1)], data[str(column_2)])
    elif statistical_test == "Wilcoxon rank-sum statistic":
        statistic, p_value = ranksums(data[str(column_1)], data[str(column_2)], nan_policy=nan_policy, alternative=alternative)
    elif statistical_test == "one-way chi-square test":
        statistic, p_value = chisquare(data[str(column_1)], data[str(column_2)])
    return {'mean_positive': np.mean(data[str(column_1)]), 'standard_deviation_positive': np.std(data[str(column_1)]),
            'mean_negative': np.mean(data[str(column_2)]), 'standard_deviation_negative': np.std(data[str(column_2)]),
            'statistic': statistic, 'p-value': p_value}


@router.post("/multiple_comparisons", tags=['hypothesis_testing'])
async def p_value_correction(input_config: ModelMultipleComparisons):
    method = input_config.method
    alpha = input_config.alpha
    p_value = input_config.p_value

    if method == 'Bonferroni':
        z = multipletests(pvals=p_value, alpha=alpha, method='bonferroni')
        y = [str(x) for x in z[0]]
        return {'rejected': list(y), 'corrected_p_values': list(z[1])}
    elif method == 'sidak':
        z = multipletests(pvals=p_value, alpha=alpha, method='sidak')
        y = [str(x) for x in z[0]]
        return {'rejected': list(y), 'corrected_p_values': list(z[1])}
    elif method == 'benjamini-hochberg':
        z = multipletests(pvals=p_value, alpha=alpha, method='fdr_bh')
        y = [str(x) for x in z[0]]
        return {'rejected': list(y), 'corrected_p_values': list(z[1])}
    elif method == 'benjamini-yekutieli':
        z = multipletests(pvals=p_value, alpha=alpha, method='fdr_by')
        y = [str(x) for x in z[0]]
        return {'rejected': list(y), 'corrected_p_values': list(z[1])}
    else:
        z = multipletests(pvals=p_value, alpha=alpha, method= method)
        y = [str(x) for x in z[0]]
        return {'rejected': list(y), 'corrected_p_values': list(z[1])}

@router.get("/classification_for_cutoff_determination")
async def classification_cutoff_determination(dependent_variable: str,
                                              independent_variables: list[str] | None = Query(default=None),
                                              algorithm: str | None = Query("SVM",
                                                                            regex="^(SVM)$|^(LDA)$")):
    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    if algorithm=='SVM':
        clf = SVC(kernel='linear')
    else:
        clf = LinearDiscriminantAnalysis()

    clf.fit(X, Y)
    coeffs = np.squeeze(clf.coef_)
    inter = clf.intercept_
    return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist()}


@router.get("/principal_component_analysis")
async def principal_component_analysis(n_components: int ,
                                       independent_variables: list[str] | None = Query(default=None)):
    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)

    pca = PCA(n_components=n_components)
    pca.fit(X)

    return {'Percentage of variance explained by each of the selected components': pca.explained_variance_ratio_.tolist(),
            'The singular values corresponding to each of the selected components. ': pca.singular_values_.tolist(),
            'Principal axes in feature space, representing the directions of maximum variance in the data.' : pca.components_.tolist()}

@router.get("/kmeans_clustering")
async def kmeans_clustering(n_clusters: int ,
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

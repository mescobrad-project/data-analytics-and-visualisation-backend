import numpy as np
import pandas as pd
from scipy.stats import shapiro, kstest, anderson, normaltest, boxcox, yeojohnson, bartlett, levene, fligner, obrientransform, pearsonr, pointbiserialr, ttest_ind, mannwhitneyu, wilcoxon,ttest_rel
from typing import Optional, Union, List
from statsmodels.stats.multitest import multipletests
from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI, Path, Query, APIRouter

router = APIRouter()
data = pd.read_csv('example_data/mescobrad_dataset.csv')

@router.get("/return_columns")
async def name_columns():
    columns = data.columns
    return{'columns': list(columns)}


@router.get("/normality_tests")
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

@router.get("/compute_point_biserial_correlation")
async def pearson_correlation(column_1: str, column_2: str):
    pointbiserialr_test = pointbiserialr(data[str(column_1)], data[str(column_2)])
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
async def statistical_tests(column1: str,
                            correction: bool,
                            statistical_test: str | None = Query("Independent t-test",
                                                                 regex="^(Independent t-test)$|^(Welch’s t-test)$|^(Mann-Whitney U rank test)$|^(t-test on TWO RELATED samples of scores)$|^(Wilcoxon signed-rank test)$"),
                            alternative: Optional[str] | None = Query("two-sided",
                                                                      regex="^(two-sided)$|^(less)$|^(greater)$"),
                            method: Optional[str] | None = Query("auto",
                                                                 regex="^(auto)$|^(asymptotic)$|^(exact)$"),
                            mode: Optional[str] | None = Query("auto",
                                                                 regex="^(auto)$|^(approx)$|^(exact)$"),
                            zero_method: Optional[str] | None = Query("pratt",
                                                                 regex="^(pratt)$|^(wilcox)$|^(zsplit)$")):
    statistic = None
    p_value = None
    positive_data = []
    negative_data = []
    for i in range(len(data)):
        if data.iloc[i]['label'] == 1:
            positive_data.append(data.iloc[i][str(column1)])
        else:
            negative_data.append(data.iloc[i][str(column1)])

    if statistical_test == "Welch's t-test":
        statistic, p_value = ttest_ind(positive_data, negative_data, equal_var=False, alternative=alternative)
    elif statistical_test == "Independent t-test":
        statistic, p_value = ttest_ind(positive_data, negative_data, alternative=alternative)
    elif statistical_test == "t-test on TWO RELATED samples of scores":
        if np.shape(positive_data)[0] != np.shape(negative_data)[0]:
            return {'error': 'Unequal length arrays'}
        statistic, p_value = ttest_rel(positive_data, negative_data, alternative=alternative)
    elif statistical_test == "Mann-Whitney U rank test":
        statistic, p_value = mannwhitneyu(positive_data, negative_data, alternative=alternative, method=method)
    elif statistical_test == "Wilcoxon signed-rank test":
        if np.shape(positive_data)[0] != np.shape(negative_data)[0]:
            return {'error': 'Unequal length arrays'}
        statistic, p_value = wilcoxon(positive_data, negative_data, alternative=alternative, correction=correction, zero_method=zero_method, mode=mode)
    return {'mean_positive': np.mean(positive_data), 'standard_deviation_positive': np.std(positive_data),
            'mean_negative': np.mean(negative_data), 'standard_deviation_negative': np.std(negative_data),
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


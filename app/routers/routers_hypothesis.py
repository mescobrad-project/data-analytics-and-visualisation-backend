import numpy as np
import pandas as pd
import json
from scipy.stats import jarque_bera, fisher_exact, ranksums, chisquare, kruskal, alexandergovern, kendalltau, f_oneway, shapiro, \
    kstest, anderson, normaltest, boxcox, yeojohnson, bartlett, levene, fligner, obrientransform, pearsonr, spearmanr, \
    pointbiserialr, ttest_ind, mannwhitneyu, wilcoxon, ttest_rel, skew, kurtosis, probplot, zscore
from typing import Optional, Union, List
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import matplotlib.pyplot as plt
import mpld3
from enum import Enum
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
from pydantic import BaseModel
from statsmodels.stats.diagnostic import het_goldfeldquandt
from fastapi import FastAPI, Path, Query, APIRouter
import sklearn
import pingouin
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from lifelines.utils import to_episodic_format
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, SGDClassifier, HuberRegressor,Lars, PoissonRegressor
from sklearn.svm import SVR, LinearSVR, LinearSVC
from pingouin import ancova
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import mpld3
from statsmodels.stats.anova import AnovaRM
from lifelines import KaplanMeierFitter, CoxTimeVaryingFitter
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.discrete.conditional_models import ConditionalLogit
from zepid.base import RiskRatio, RiskDifference, OddsRatio, IncidenceRateRatio, IncidenceRateDifference, NNT
from zepid import load_sample_data
from zepid.calc import risk_ci, incidence_rate_ci, risk_ratio, risk_difference, number_needed_to_treat, odds_ratio, incidence_rate_ratio, incidence_rate_difference
from app.pydantic_models import ModelMultipleComparisons
from app.utils.utils_datalake import fget_object, get_saved_dataset_for_Hypothesis
from app.utils.utils_general import get_local_storage_path, get_single_file_from_local_temp_storage, load_data_from_csv, \
    load_file_csv_direct
import scipy.stats as st
import statistics
from tabulate import tabulate

router = APIRouter()
data = pd.read_csv('example_data/mescobrad_dataset.csv')
data = data.drop(["Unnamed: 0"], axis=1)
# data = pd.read_csv('example_data/sample_questionnaire.csv')

def normality_test_content_results(column: str, selected_dataframe):
    if (column):
        # region Creating Box-plot
        fig2 = plt.figure()
        plt.boxplot(selected_dataframe[str(column)])
        plt.ylabel("", fontsize=14)
        # show plot
        plt.show()
        html_str_B = mpld3.fig_to_html(fig2)
        #endregion
        # region Creating QQ-plot
        fig = sm.qqplot(selected_dataframe[str(column)], line='45')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        html_str = mpld3.fig_to_html(fig)
        # endregion
        # region Creating Probability-plot
        fig3 = plt.figure()
        ax1 = fig3.add_subplot()
        prob =  probplot(selected_dataframe[str(column)], dist=st.norm, plot=ax1)
        ax1.set_title('Probplot against normal distribution')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        html_str_P = mpld3.fig_to_html(fig3)
        # endregion
        #region Creating histogram
        fig1, axs = plt.subplots(1, 1,
                                 # figsize=(640, 480),
                                 tight_layout=True)

        ## q25, q75 = np.percentile(data[str(column)], [25, 75])
        ## bin_width = 2 * (q75 - q25) * len(data[str(column)]) ** (-1 / 3)
        ## bins = round((data[str(column)].max() - data[str(column)].min()) / bin_width)
        axs.hist(selected_dataframe[str(column)], density=True, bins=30, label="Data", rwidth=0.9,
                 color='#607c8e')

        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 300)
        kde = st.gaussian_kde(selected_dataframe[str(column)])
        plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
        plt.legend(loc="upper left")
        plt.ylabel("Probability", fontsize=14)
        plt.xlabel("Data", fontsize=14)
        plt.title("Histogram", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        html_str_H = mpld3.fig_to_html(fig1)
        #endregion
        #region Calculate skew, kurtosis, median, std, etc.
        skewtosend = skew(selected_dataframe[str(column)], axis=0, bias=True)
        kurtosistosend = kurtosis(selected_dataframe[str(column)], axis=0, bias=True)
        st_dev = np.std(selected_dataframe[str(column)])
        # Used Statistics lib for cross-checking
        # standard_deviation = statistics.stdev(data[str(column)])
        median_value = float(np.percentile(selected_dataframe[str(column)], 50))
        # Used a different way to calculate Median
        # TODO: we must investigate why it returns a different value
        # med2 = np.median(data[str(column)])
        mean_value = np.mean(selected_dataframe[str(column)])
        num_rows = selected_dataframe[str(column)].shape
        top5 = sorted(selected_dataframe[str(column)].tolist(), reverse=True)[:5]
        last5 = sorted(selected_dataframe[str(column)].tolist(), reverse=True)[-5:]
        #endregion
        return {'plot_column': column, 'qqplot': html_str, 'histogramplot': html_str_H, 'boxplot': html_str_B, 'probplot': html_str_P, 'skew': skewtosend, 'kurtosis': kurtosistosend, 'standard_deviation': st_dev, "median": median_value, "mean": mean_value, "sample_N": num_rows, "top_5": top5, "last_5": last5}
    else:
        return {'plot_column': "", 'qqplot': "", 'histogramplot': "", 'boxplot': "", 'probplot': "",
                'skew': 0, 'kurtosis': 0,
                'standard_deviation': 0, "median": 0,
                "mean": 0, "sample_N": 0, "top_5": [], "last_5": []}

def transformation_extra_content_results(column_In: str, column_Out:str, selected_dataframe):
    fig = plt.figure()
    plt.plot(selected_dataframe[str(column_In)], selected_dataframe[str(column_In)],
             color='blue', marker="*")
    # red for numpy.log()
    plt.plot(selected_dataframe[str(column_Out)], selected_dataframe[str(column_In)],
             color='red', marker="o")
    plt.title("Transformed data Comparison")
    plt.xlabel("out_array")
    plt.ylabel("in_array")
    plt.show()
    html_str_Transf = mpld3.fig_to_html(fig)
    return html_str_Transf

@router.get("/load_demo_data")
async def load_demo_data(file: Optional[str] | None):
    # print(file, len(file))
    # file = None
    if file!= None:
        data = pd.read_csv('runtime_config/' + file)
    else:
        data = pd.read_csv('example_data/sample_questionnaire.csv')

    return data

@router.get("/return_columns")
async def name_columns(step_id: str, run_id: str):
    path_to_storage = get_local_storage_path(run_id, step_id)
    name_of_file = get_single_file_from_local_temp_storage(run_id, step_id)
    data = load_data_from_csv(path_to_storage + "/" + name_of_file)

    columns = data.columns
    return{'columns': list(columns)}

@router.get("/return_saved_object_columns")
async def name_saved_object_columns(file_name:str):
    print('saved', file_name, 'runtime_config/' + file_name)
    try:
        get_saved_dataset_for_Hypothesis('saved', file_name, 'runtime_config/'+file_name)
        data = pd.read_csv('runtime_config/'+file_name)
        columns = data.columns
        return{'columns': list(columns)}
    except:
        return{'columns': {}}

@router.get("/normality_tests", tags=['hypothesis_testing'])
async def normal_tests(step_id: str, run_id: str,
                       column: str,
                       nan_policy: Optional[str] | None = Query("propagate",
                                                                regex="^(propagate)$|^(raise)$|^(omit)$"),
                       axis: Optional[int] = 0,
                       alternative: Optional[str] | None = Query("two-sided",
                                                                 regex="^(two-sided)$|^(less)$|^(greater)$"),
                       name_test: str | None = Query("Shapiro-Wilk",
                                                   regex="^(Shapiro-Wilk)$|^(Kolmogorov-Smirnov)$|^(Anderson-Darling)$|^(D’Agostino’s K\^2)$|^(Jarque-Bera)$")) -> dict:

    data = load_file_csv_direct(run_id, step_id)
    results_to_send = normality_test_content_results(column, data)

    # region AmCharts_CODE_REGION
    # # ******************************************
    # # Data where prepared for Amcharts but now are not needed
    # # for BoxPlot chart
    # # 'min,q1,median,q3,max'
    # boxplot_data = [{
    #     "date": "2022-05-18",
    #     # "name": column,
    #     "min": float(np.min(data[str(column)])),
    #     "q1": float(np.percentile(data[str(column)], 25)),
    #     "q2": float(np.percentile(data[str(column)], 50)),
    #     "q3": float(np.percentile(data[str(column)], 75)),
    #     "max": float(np.max(data[str(column)]))
    # }]
    # # ******************************************
    # endregion

    if name_test == 'Shapiro-Wilk':
        shapiro_test = shapiro(data[str(column)])
        if shapiro_test.pvalue > 0.05:
            return{'statistic': shapiro_test.statistic, 'p_value': shapiro_test.pvalue, 'Description': 'Sample looks Gaussian (fail to reject H0)', 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
        else:
            return{'statistic': shapiro_test.statistic, 'p_value': shapiro_test.pvalue, 'Description': 'Sample does not look Gaussian (reject H0)', 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
    elif name_test == 'Kolmogorov-Smirnov':
        ks_test = kstest(data[str(column)], 'norm', alternative=alternative)
        if ks_test.pvalue > 0.05:
            return{'statistic': ks_test.statistic, 'p_value': ks_test.pvalue, 'Description':'Sample looks Gaussian (fail to reject H0)', 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
        else:
            return{'statistic': ks_test.statistic, 'p_value': ks_test.pvalue, 'Description':'Sample does not look Gaussian (reject H0)', 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
    elif name_test == 'Anderson-Darling':
        anderson_test = anderson(data[str(column)])
        list_anderson = []
        for i in range(len(anderson_test.critical_values)):
            sl, cv = anderson_test.significance_level[i], anderson_test.critical_values[i]
            if anderson_test.statistic < anderson_test.critical_values[i]:
                # print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
                list_anderson.append('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
            else:
                # print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
                list_anderson.append('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
        return{'statistic':anderson_test.statistic, 'critical_values': list(anderson_test.critical_values), 'significance_level': list(anderson_test.significance_level), 'Description': list_anderson, 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
    elif name_test == 'D’Agostino’s K^2':
        stat, p = normaltest(data[str(column)], nan_policy=nan_policy)
        if p > 0.05:
            return{'statistic': stat, 'p_value': p, 'Description':'Sample looks Gaussian (fail to reject H0)', 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
        else:
            return{'statistic': stat, 'p_value': p, 'Description':'Sample does not look Gaussian (reject H0)', 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
    elif name_test == 'Jarque-Bera':
        jarque_bera_test = jarque_bera(data[str(column)])
        statistic = jarque_bera_test.statistic
        pvalue = jarque_bera_test.pvalue
        if pvalue > 0.05:
            return {'statistic': statistic, 'p_value': pvalue,
                    'Description': 'Sample looks Gaussian (fail to reject H0)', 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
        else:
            return {'statistic': statistic, 'p_value': pvalue,
                    'Description': 'Sample does not look Gaussian (reject H0)', 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}


@router.get("/transform_data", tags=['hypothesis_testing'])
async def transform_data(step_id: str,
                         run_id: str,
                         column: str,
                         name_transform: str | None = Query("Box-Cox",
                                                           regex="^(Box-Cox)$|^(Yeo-Johnson)$|^(Log)$|^(Squared-root)$|^(Cube-root)$"),
                         lmbd: Optional[float] = None,
                         alpha: Optional[float] = None) -> dict:

    data = load_file_csv_direct(run_id, step_id)
    newColumnName = "Transf_" + column
    if name_transform == 'Box-Cox':
        if lmbd == None:
            if alpha == None:
                boxcox_array, maxlog = boxcox(np.array(data[str(column)]))
                data[newColumnName] = boxcox_array
                results_to_send = normality_test_content_results(newColumnName, data)
                results_to_send['transf_plot'] = transformation_extra_content_results(column, newColumnName, data)
                return {'Box-Cox power transformed array': list(boxcox_array), 'lambda that maximizes the log-likelihood function': maxlog, 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
            else:
                boxcox_array, maxlog, z = boxcox(np.array(data[str(column)]), alpha=alpha)
                data[newColumnName] = boxcox_array
                results_to_send = normality_test_content_results(newColumnName, data)
                results_to_send['transf_plot'] = transformation_extra_content_results(column, newColumnName, data)
                return {'Box-Cox power transformed array': list(boxcox_array), 'lambda that maximizes the log-likelihood function': maxlog, 'minimum confidence limit': z[0], 'maximum confidence limit': z[1], 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
        else:
            if alpha == None:
                y = boxcox(np.array(data[str(column)]), lmbda=lmbd)
                data[newColumnName] = y
                results_to_send = normality_test_content_results(newColumnName, data)
                results_to_send['transf_plot'] = transformation_extra_content_results(column, newColumnName, data)
                return {'Box-Cox power transformed array': list(y), 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
            else:
                y = boxcox(np.array(data[str(column)]), lmbda=lmbd, alpha=alpha)
                data[newColumnName] = y
                results_to_send = normality_test_content_results(newColumnName, data)
                results_to_send['transf_plot'] = transformation_extra_content_results(column, newColumnName, data)
                return {'Box-Cox power transformed array': list(y), 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
    elif name_transform == 'Yeo-Johnson':
        if lmbd == None:
            yeojohnson_array, maxlog = yeojohnson(np.array(data[str(column)]))
            data[newColumnName] = yeojohnson_array
            results_to_send = normality_test_content_results(newColumnName, data)
            results_to_send['transf_plot'] = transformation_extra_content_results(column, newColumnName, data)
            return {'Yeo-Johnson power transformed array': list(yeojohnson_array), 'lambda that maximizes the log-likelihood function': maxlog, 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
        else:
            yeojohnson_array = yeojohnson(np.array(data[str(column)]), lmbda=lmbd)
            data[newColumnName] = yeojohnson_array
            results_to_send = normality_test_content_results(newColumnName, data)
            results_to_send['transf_plot'] = transformation_extra_content_results(column, newColumnName, data)
            return {'Yeo-Johnson power transformed array': list(yeojohnson_array), 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
    elif name_transform == 'Log':
        log_array = np.log(data[str(column)])
        data[newColumnName] = log_array
        results_to_send = normality_test_content_results(newColumnName, data)
        results_to_send['transf_plot'] = transformation_extra_content_results(column, newColumnName, data)
        return {'Log transformed array': list(log_array), 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
    elif name_transform == 'Squared-root':
        sqrt_array = np.sqrt(data[str(column)])
        data[newColumnName] = sqrt_array
        results_to_send = normality_test_content_results(newColumnName, data)
        results_to_send['transf_plot'] = transformation_extra_content_results(column, newColumnName, data)
        return {'Squared-root transformed array': list(sqrt_array), 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
    elif name_transform == 'Cube-root':
        cbrt_array = np.cbrt(data[str(column)])
        data[newColumnName] = cbrt_array
        results_to_send = normality_test_content_results(newColumnName, data)
        results_to_send['transf_plot'] = transformation_extra_content_results(column, newColumnName, data)
        return {'Cube-root transformed array': list(cbrt_array), 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}


@router.get("/compute_pearson_correlation", tags=['hypothesis_testing'])
async def pearson_correlation(step_id: str, run_id: str, column_1: str, column_2: str):
    data = load_file_csv_direct(run_id, step_id)
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
async def point_biserial_correlation(step_id: str, run_id: str, column_1: str, column_2: str):
    data = load_file_csv_direct(run_id, step_id)
    unique_values = np.unique(data[str(column_1)])
    if len(unique_values) == 2:
        pointbiserialr_test = pointbiserialr(data[str(column_1)], data[str(column_2)])
    else:
        pointbiserialr_test = pointbiserialr(data[str(column_2)], data[str(column_1)])
    return {'correlation':pointbiserialr_test[0], 'p-value': pointbiserialr_test[1]}

#
@router.get("/check_homoscedasticity", tags=['hypothesis_testing'])
async def check_homoskedasticity(step_id: str,
                                 run_id: str,
                                 columns: list[str] | None = Query(default=None),
                                 name_of_test: str | None = Query("Levene",
                                                                  regex="^(Levene)$|^(Bartlett)$|^(Fligner-Killeen)$"),
                                 center: Optional[str] | None = Query("median",
                                                                      regex="^(trimmed)$|^(median)$|^(mean)$")):
    data = load_file_csv_direct(run_id, step_id)

    args = []
    var = []
    i = 0
    for k in columns:
        args.append(data[k])
        temp_to_append = {
            "id": i,
            "Variable": k,
            "Variance": np.var(data[k], ddof=0)
        }
        var.append(temp_to_append)
        i = i + 1

    if name_of_test == "Bartlett":
        statistic, p_value = bartlett(*args)
    elif name_of_test == "Fligner-Killeen":
        statistic, p_value = fligner(*args, center=center)
    else:
        statistic, p_value = levene(*args, center=center)
    return {'statistic': statistic, 'p_value': p_value, 'variance': var}


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


@router.get("/return_LDA", tags=["return_LDA"])
async def LDA(dependent_variable: str,
              solver: str | None = Query("svd",
                                         regex="^(svd)$|^(lsqr)$|^(eigen)$"),
              shrinkage_1: str | None = Query("none",
                                              regex="^(none)$|^(auto)$"),
              shrinkage_2: float | None = Query(default=None, gt=0, lt=0),
              shrinkage_3: float | None = Query(default=None),
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

    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_json(orient='split')}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_json(orient='split')}

@router.get("/SVC_function")
async def SVC_function(dependent_variable: str,
                       degree: int | None = Query(default=3),
                       max_iter: int | None = Query(default=-1),
                       C: float | None = Query(default=1,gt=0),
                       coef0: float | None = Query(default=0),
                       gamma: str | None = Query("scale",
                                                 regex="^(scale)$|^(auto)$"),
                       kernel: str | None = Query("rbf",
                                                  regex="^(rbf)$|^(linear)$|^(poly)$|^(sigmoid)$"),
                       independent_variables: list[str] | None = Query(default=None)):

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    if kernel == 'poly':
        clf = SVC(degree=degree, kernel=kernel, gamma=gamma, coef0=coef0, C=C, max_iter=max_iter)
    elif kernel == 'rbf' or kernel == 'sigmoid':
        if kernel == 'sigmoid':
            clf = SVC(gamma=gamma, kernel=kernel, coef0=coef0, C=C, max_iter=max_iter)
        else:
            clf = SVC(gamma=gamma, C=C, kernel=kernel, max_iter=max_iter)
    else:
        clf = SVC(kernel=kernel, C=C, max_iter=max_iter)


    clf.fit(X, Y)

    if kernel == 'linear':
        if np.shape(X)[1] == 1:
            coeffs = clf.coef_
            inter = clf.intercept_
            df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
            df_names = pd.DataFrame(dataset.columns, columns=['variables'])
            df = pd.concat([df_names, df_coeffs], axis=1)
            return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_json(orient='split')}
        else:
            coeffs = np.squeeze(clf.coef_)
            inter = clf.intercept_
            df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
            df_names = pd.DataFrame(dataset.columns, columns=['variables'])
            df = pd.concat([df_names, df_coeffs], axis=1)
            return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                    'dataframe': df.to_json(orient='split')}
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
    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_json(orient='split')}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_json(orient='split')}

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
    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_json(orient='split')}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_json(orient='split')}

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
    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_json(orient='split')}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_json(orient='split')}

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
    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_json(orient='split')}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_json(orient='split')}

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
    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_json(orient='split')}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_json(orient='split')}

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
    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        outliers = clf.outliers_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'outliers':outliers.tolist(), 'dataframe': df.to_json(orient='split')}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        outliers = clf.outliers_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'outliers':outliers.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_json(orient='split')}


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
                                                 regex="^(rbf)$|^(linear)$|^(poly)$|^(sigmoid)$"),

                        independent_variables: list[str] | None = Query(default=None)):

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    if kernel == 'poly':
        clf = SVR(degree=degree, kernel=kernel, gamma=gamma, coef0=coef0, C=C, epsilon=epsilon, max_iter=max_iter)
    elif kernel == 'rbf' or kernel == 'sigmoid':
        if kernel == 'sigmoid':
            clf = SVR(gamma=gamma, kernel=kernel, coef0=coef0, C=C, epsilon=epsilon, max_iter=max_iter)
        else:
            clf = SVR(gamma=gamma, kernel=kernel, C=C, epsilon=epsilon, max_iter=max_iter)
    else:
        clf = SVR(C=C, kernel=kernel, epsilon=epsilon, max_iter=max_iter)

    clf.fit(X, Y)
    if kernel == 'linear':
        if np.shape(X)[1] == 1:
            coeffs = clf.coef_
            inter = clf.intercept_
            df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
            df_names = pd.DataFrame(dataset.columns, columns=['variables'])
            df = pd.concat([df_names, df_coeffs], axis=1)
            return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                    'dataframe': df.to_json(orient='split')}
        else:
            coeffs = np.squeeze(clf.coef_)
            inter = clf.intercept_
            df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
            df_names = pd.DataFrame(dataset.columns, columns=['variables'])
            df = pd.concat([df_names, df_coeffs], axis=1)
            return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                    'dataframe': df.to_json(orient='split')}
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
    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_json(orient='split')}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_json(orient='split')}


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
    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_json(orient='split')}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_json(orient='split')}

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

    df_0 = df.tables[0]
    df_1 = df.tables[1]

    return {'first table': df_0.to_json(orient='split'), 'second table': df_1.to_json(orient='split')}

@router.get("/poisson_regression")
async def poisson_regression(dependent_variable: str,
                             alpha: float | None = Query(default=1.0, ge=0),
                             max_iter: int | None = Query(default=1000),
                             independent_variables: list[str] | None = Query(default=None)):

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')

    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label)

    clf = PoissonRegressor(alpha=alpha, max_iter=max_iter)

    clf.fit(X, Y)
    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_json(orient='split')}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_json(orient='split')}

@router.get("/cox_regression")
async def cox_regression(duration_col: str,
                         covariates: str,
                         alpha: float | None = Query(default=0.05),
                         penalizer: float | None = Query(default=0.0),
                         l1_ratio: float | None = Query(default=0.0),
                         n_baseline_knots: int | None = Query(default=2),
                         breakpoints: int | None = Query(default=None),
                         event_col: str | None = Query(default=None),
                         weights_col: str | None = Query(default=None),
                         cluster_col: str | None = Query(default=None),
                         entry_col: str | None = Query(default=None),
                         strata: list[str] | None = Query(default=None),
                         values: list[int] | None = Query(default=None),
                         hazard_ratios: bool | None = Query(default=False),
                         baseline_estimation_method: str | None = Query("breslow",
                                                                       regex="^(breslow)$|^(spline)$|^(piecewise)$")):

    to_return = {}

    fig = plt.figure(1)
    ax = plt.subplot(111)

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')

    if baseline_estimation_method == "spline":
        cph = CoxPHFitter(alpha=alpha, baseline_estimation_method=baseline_estimation_method,penalizer=penalizer,l1_ratio=l1_ratio,strata=strata,
                          n_baseline_knots=n_baseline_knots)
    elif baseline_estimation_method == "piecewise":
        cph = CoxPHFitter(alpha=alpha, baseline_estimation_method=baseline_estimation_method, penalizer=penalizer, l1_ratio=l1_ratio,strata=strata,
                          breakpoints=breakpoints)
    else:
        cph = CoxPHFitter(alpha=alpha,baseline_estimation_method=baseline_estimation_method, penalizer=penalizer, l1_ratio=l1_ratio,strata=strata)

    cph.fit(dataset, duration_col=duration_col, event_col=event_col,weights_col=weights_col,cluster_col=cluster_col,entry_col=entry_col)

    df = cph.summary

    #fig = plt.figure(figsize=(18, 12))
    cph.plot(hazard_ratios=hazard_ratios, ax=ax)
    plt.show()

    html_str = mpld3.fig_to_html(fig)
    to_return["figure_1"] = html_str
    #plt.close(1)
    #fig = plt.figure(2)
    #ax = plt.subplot(121)
    plt.clf()
    if values!=None:
        cph.plot_partial_effects_on_outcome(covariates=covariates, values=values, cmap='coolwarm')
        plt.show()
        html_str = mpld3.fig_to_html(fig)
        to_return["figure_2"] = html_str

    results = proportional_hazard_test(cph, dataset, time_transform='rank')

    df_1 = results.summary
    AIC = cph.AIC_partial_
    return {'Concordance Index':cph.concordance_index_ ,'Akaike information criterion (AIC) (partial log-likelihood)': AIC,'Dataframe of the coefficients, p-values, CIs, etc.':df.to_json(orient="split"), 'figure': to_return, 'proportional hazard test': df_1.to_json(orient='split')}

@router.get("/time_varying_covariates")
async def time_varying_covariates(event_col: str,
                                  duration_col:str,
                                  column_1:str | None = Query(default=None),
                                  column_2:str | None = Query(default=None),
                                  correction_columns: bool | None = Query(default=False),
                                  time_gaps: float | None = Query(default=1.),
                                  alpha: float | None = Query(default=0.05),
                                  penalizer: float | None = Query(default=0.0),
                                  l1_ratio: float | None = Query(default=0.0),
                                  weights_col: str | None = Query(default=None),
                                  strata: list[str] | None = Query(default=None)):

    to_return = {}

    fig = plt.figure(1)
    ax = plt.subplot(111)

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')

    dataset_long = to_episodic_format(dataset, duration_col=duration_col, event_col=event_col, time_gaps=time_gaps)

    if correction_columns:
        dataset_long[column_1+'*'+column_2] = dataset_long[column_1]*dataset_long[column_2]

    cph = CoxTimeVaryingFitter(alpha=alpha, penalizer=penalizer, l1_ratio=l1_ratio)

    cph.fit(dataset_long, event_col=event_col, id_col='id', weights_col=weights_col,start_col='start', stop_col='stop',strata=strata)

    df = cph.summary

    #fig = plt.figure(figsize=(18, 12))
    cph.plot(ax=ax)
    plt.show()

    html_str = mpld3.fig_to_html(fig)
    to_return["figure_1"] = html_str

    return {'Akaike information criterion (AIC) (partial log-likelihood)':cph.AIC_partial_,'Dataframe of the coefficients, p-values, CIs, etc.':df.to_json(orient="split"), 'figure': to_return}

@router.get("/anova_repeated_measures")
async def anova_rm(dependent_variable: str,
                   subject: str,
                   within: list[str] | None = Query(default=None),
                   aggregate_func: str | None = Query(default=None,
                                                      regex="^(mean)$")):

    df_data = pd.read_csv('example_data/mescobrad_dataset.csv')

    unique, counts = np.unique(df_data[subject], return_counts=True)

    z = all(x==counts[0] for x in counts)
    if z:
        df = AnovaRM(data=df_data, depvar=dependent_variable, subject=subject, within=within, aggregate_func=aggregate_func)
        df_new = df.fit()
        return{'Result': df_new}
    else:
        return {"Unbalanced"}

@router.get("/generalized_estimating_equations")
async def generalized_estimating_equations(dependent: str,
                                           groups: str,
                                           independent: list[str] | None = Query(default=None),
                                           conv_struct: str | None = Query("independence",
                                                                           regex="^(independence)$|^(autoregressive)$|^(exchangeable)$|^(nested_working_dependence)$"),
                                           family: str | None = Query("poisson",
                                                                      regex="^(poisson)$|^(gamma)$|^(gaussian)$|^(inverse_gaussian)$|^(negative_binomial)$|^(binomial)$|^(tweedie)$")):

    data = pd.read_csv('example_data/mescobrad_dataset.csv')

    z = dependent + "~"
    for i in range(len(independent)):
        z = z + "+" + independent[i]

    if family == "poisson":
        fam = sm.families.Poisson()
    elif family == "gamma":
        fam = sm.families.Gamma()
    elif family == "gaussian":
        fam = sm.families.Gaussian()
    elif family == "inverse_gaussian":
        fam = sm.families.InverseGaussian()
    elif family == 'negative_binomial':
        fam = sm.families.NegativeBinomial()
    elif family == "binomial":
        fam = sm.families.Binomial()
    else:
        fam = sm.families.Tweedie()

    if conv_struct == "independence":
        ind = sm.cov_struct.Independence()
    elif conv_struct == "autoregressive":
        ind = sm.cov_struct.Autoregressive()
    elif conv_struct == "exchangeable":
        ind = sm.cov_struct.Exchangeable()
    else:
        ind = sm.cov_struct.Nested()

    md = smf.gee(z, groups, data, cov_struct=ind, family=fam)

    mdf = md.fit()

    df = mdf.summary()

    print(df)

    results_as_html = df.tables[0].as_html()
    df_0 = pd.read_html(results_as_html)[0]

    results_as_html = df.tables[1].as_html()
    df_1 = pd.read_html(results_as_html)[0]

    results_as_html = df.tables[2].as_html()
    df_2 = pd.read_html(results_as_html)[0]

    return {'first_table':df_0.to_json(orient="split"), 'second table':df_1.to_json(orient="split"), 'third table':df_2.to_json(orient="split")}

@router.get("/kaplan_meier")
async def kaplan_meier(column_1: str,
                       column_2: str,
                       at_risk_counts: bool | None = Query(default=True),
                       label: str | None = Query(default=None),
                       alpha: float | None = Query(default=0.05)):
    to_return = {}

    fig = plt.figure(1)
    ax = plt.subplot(111)

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')

    kmf = KaplanMeierFitter(alpha=alpha, label=label)
    kmf.fit(dataset[column_1], dataset[column_2])
    kmf.plot_survival_function(at_risk_counts=at_risk_counts)
    plt.show()

    html_str = mpld3.fig_to_html(fig)
    to_return["figure_1"] = html_str

    df = kmf.survival_function_
    confidence_interval = kmf.confidence_interval_

    return {'figure': to_return, "survival_function":df.to_json(orient="split"), "confidence_interval": confidence_interval.to_json(orient='split')}

@router.get("/fisher")
async def fisher(variable_top_left: int,
                 variable_top_right: int,
                 variable_bottom_left: int,
                 variable_bottom_right: int,
                 alternative: Optional[str] | None = Query("two-sided",
                                                           regex="^(two-sided)$|^(less)$|^(greater)$")):

    df = [[variable_top_left,variable_top_right], [variable_bottom_left,variable_bottom_right]]

    odd_ratio, p_value = fisher_exact(df, alternative=alternative)

    return {'odd_ratio': odd_ratio, "p_value": p_value}

@router.get("/mc_nemar")
async def mc_nemar(variable_top_left: int,
                   variable_top_right: int,
                   variable_bottom_left: int,
                   variable_bottom_right: int,
                   exact: bool | None = Query(default=False),
                   correction: bool | None = Query(default=True)):

    df = [[variable_top_left,variable_top_right], [variable_bottom_left,variable_bottom_right]]

    result = mcnemar(df, exact=exact, correction=correction)

    return {'statistic': result.statistic, "p_value": result.pvalue}

@router.get("/all_statistics")
async def all_statistics():

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    print(dataset.describe())

    return {'statistics':dataset.to_json(orient="split")}

@router.get("/conditional_logistic_regression")
async def conditional_logistic_regression(endog: str,
                                          exog: str,
                                          groups: str,
                                          method: str | None = Query("bfgs",
                                                                     regex="^(bfgs)$|^(newton)$|^(lbfgs)$|^(powell)$|^(cg)$|^(ncg)$|^(basinhopping)$|^(minimize)$")
                                          ):

    dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    z = np.unique(dataset[endog])
    if len(z) == 2 and 0 in z and 1 in z:
        m = ConditionalLogit(endog=dataset[endog], exog=dataset[exog], groups=dataset[groups])
        k = m.fit(method=method)
        df = k.summary()
        print(df)

        results_as_html = df.tables[0].as_html()
        df_0 = pd.read_html(results_as_html)[0]

        results_as_html = df.tables[1].as_html()
        df_1 = pd.read_html(results_as_html)[0]

        return {'first_table': df_0.to_json(orient="split"), 'second table': df_1.to_json(orient="split")}
    else:
        return {'The response variable must contain only 0 and 1'}


@router.get("/risks")
async def risk_ratio_1(exposure: str,
                       outcome: str,
                       time: str | None = Query(default=None),
                       reference: int | None = Query(default=0),
                       alpha: float | None = Query(default=0.05),
                       method: str | None = Query("risk_ratio",
                                                  regex="^(risk_ratio)$|^(risk_difference)$|^(number_needed_to_treat)$|^(odds_ratio)$|^(incidence_rate_ratio)$|^(incidence_rate_difference)$")):

    to_return = {}

    fig = plt.figure(1)
    ax = plt.subplot(111)

    #dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    dataset = load_sample_data(False)

    if method == 'risk_ratio':
        rr = RiskRatio(reference=reference, alpha=alpha)
    elif method == 'risk_difference':
        rr = RiskDifference(reference=reference, alpha=alpha)
    elif method == 'number_needed_to_treat':
        rr = NNT(reference=reference, alpha=alpha)
        rr.fit(dataset, exposure='art', outcome='dead')
        df = rr.results
        return {'table': df.to_json(orient="split")}
    elif method == 'odds_ratio':
        rr = OddsRatio(reference=reference, alpha=alpha)
    elif method == 'incidence_rate_ratio':
        rr = IncidenceRateRatio(reference=reference, alpha=alpha)
        rr.fit(dataset, exposure='art', outcome='dead', time='t')
        df = rr.results
        rr.plot()
        plt.show()

        html_str = mpld3.fig_to_html(fig)
        to_return["figure"] = html_str

        return {'table': df.to_json(orient="split"), 'figure': to_return}
    else:
        rr = IncidenceRateDifference(reference=reference, alpha=alpha)
        rr.fit(dataset, exposure='art', outcome='dead', time='t')
        df = rr.results
        rr.plot()
        plt.show()

        html_str = mpld3.fig_to_html(fig)
        to_return["figure"] = html_str

        return {'table': df.to_json(orient="split"), 'figure': to_return}

    rr.fit(dataset, exposure='art', outcome='dead')
    df = rr.results

    rr.plot()
    plt.show()

    html_str = mpld3.fig_to_html(fig)
    to_return["figure"] = html_str

    return {'table': df.to_json(orient="split"), 'figure': to_return}

@router.get("/two_sided_risk_ci")
async def two_sided_risk_ci(events: int,
                            total: int,
                            alpha: float | None = Query(default=0.05),
                            confint: str | None = Query("wald",
                                                       regex="^(wald)$|^(hypergeometric)$")):

    r = risk_ci(events=events, total=total, alpha=alpha, confint=confint)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error

    return {'estimated risk': estimated_risk, 'lower bound': lower_bound, 'upper bound': upper_bound, 'standard error': standard_error}

@router.get("/two_sided_incident_rate")
async def two_sided_risk_ci(events: int,
                            time: int,
                            alpha: float | None = Query(default=0.05)):

    r = incidence_rate_ci(events=events, time=time, alpha=alpha)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error

    return {'estimated incident rate': estimated_risk, 'lower bound': lower_bound, 'upper bound': upper_bound, 'standard error': standard_error}


@router.get("/risk_ratio_function")
async def risk_ratio_function(exposed_with: int,
                              unexposed_with: int,
                              exposed_without: int,
                              unexposed_without: int,
                              alpha: float | None = Query(default=0.05)):

    r = risk_ratio(a=exposed_with, b=unexposed_with, c=exposed_without, d=unexposed_without, alpha=alpha)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error

    return {'risk ratio': estimated_risk, 'lower bound': lower_bound, 'upper bound': upper_bound, 'standard error': standard_error}

@router.get("/risk_difference_function")
async def risk_difference_function(exposed_with: int,
                                   unexposed_with: int,
                                   exposed_without: int,
                                   unexposed_without: int,
                                   alpha: float | None = Query(default=0.05)):

    r = risk_difference(a=exposed_with, b=unexposed_with, c=exposed_without, d=unexposed_without, alpha=alpha)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error

    return {'risk difference': estimated_risk, 'lower bound': lower_bound, 'upper bound': upper_bound, 'standard error': standard_error}

@router.get("/number_needed_to_treat_function")
async def number_needed_to_treat_function(exposed_with: int,
                                          unexposed_with: int,
                                          exposed_without: int,
                                          unexposed_without: int,
                                          alpha: float | None = Query(default=0.05)):

    r = number_needed_to_treat(a=exposed_with, b=unexposed_with, c=exposed_without, d=unexposed_without, alpha=alpha)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error

    return {'nnt': estimated_risk, 'lower bound': lower_bound, 'upper bound': upper_bound, 'standard error': standard_error}

@router.get("/odds_ratio_function")
async def odds_ratio_function(exposed_with: int,
                              unexposed_with: int,
                              exposed_without: int,
                              unexposed_without: int,
                              alpha: float | None = Query(default=0.05)):

    r = odds_ratio(a=exposed_with, b=unexposed_with, c=exposed_without, d=unexposed_without, alpha=alpha)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error

    return {'odds ratio': estimated_risk, 'lower bound': lower_bound, 'upper bound': upper_bound, 'standard error': standard_error}

@router.get("/incidence_rate_ratio_function")
async def incidence_rate_ratio_function(exposed_with: int,
                                        unexposed_with: int,
                                        person_time_exposed: int,
                                        person_time_unexposed: int,
                                        alpha: float | None = Query(default=0.05)):

    r = incidence_rate_ratio(a=exposed_with, c=unexposed_with, t1=person_time_exposed, t2=person_time_unexposed, alpha=alpha)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error

    return {'incident rate ratio': estimated_risk, 'lower bound': lower_bound, 'upper bound': upper_bound, 'standard error': standard_error}

@router.get("/incidence_rate_difference_function")
async def incidence_rate_difference_function(exposed_with: int,
                                             unexposed_with: int,
                                             person_time_exposed: int,
                                             person_time_unexposed: int,
                                             alpha: float | None = Query(default=0.05)):

    r = incidence_rate_difference(a=exposed_with, c=unexposed_with, t1=person_time_exposed, t2=person_time_unexposed, alpha=alpha)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error

    return {'incident rate difference': estimated_risk, 'lower bound': lower_bound, 'upper bound': upper_bound, 'standard error': standard_error}

@router.get("/correlations_pingouin")
async def correlations_pingouin(column_1: str,
                                column_2: str,
                                alternative: Optional[str] | None = Query("two-sided",
                                                                          regex="^(two-sided)$|^(less)$|^(greater)$"),
                                method: Optional[str] | None = Query("pearson",
                                                                     regex="^(pearson)$|^(spearman)$|^(kendall)$|^(bicor)$|^(percbend)$|^(shepherd)$|^(skipped)$")):

    df = pingouin.corr(x=data[str(column_1)], y=data[str(column_2)], method=method, alternative=alternative)

    return {'DataFrame': df.to_json(orient='split')}

@router.get("/linear_regressor_pinguin")
async def linear_regression_pinguin(dependent_variable: str,
                                    alpha: float | None=Query(default=0.05),
                                    relimp: bool | None=Query(default=False),
                                    independent_variables: list[str] | None = Query(default=None)):

    lm = pingouin.linear_regression(data[independent_variables], data[dependent_variable], as_dataframe=True, alpha=alpha, relimp=relimp)

    return {'residuals': lm.residuals_.tolist(), 'degrees of freedom of the model': lm.df_model_, 'degrees of freedom of the residuals': lm.df_resid_ , 'dataframe': lm.to_json(orient='split')}

@router.get("/logistic_regressor_pinguin")
async def logistic_regression_pinguin(dependent_variable: str,
                                      alpha: float | None=Query(default=0.05),
                                      independent_variables: list[str] | None = Query(default=None)):

    lm = pingouin.logistic_regression(data[independent_variables], data[dependent_variable], as_dataframe=True, alpha=alpha)

    return {'dataframe': lm.to_json(orient='split')}

# @router.get("/linear_regressor_statsmodels")
# async def linear_regression_statsmodels(dependent_variable: str,
#                                         check_heteroscedasticity: bool | None = Query(default=True),
#                                         regularization: bool | None = Query(default=False),
#                                         independent_variables: list[str] | None = Query(default=None)):
#
#     x = data[independent_variables]
#     y = data[dependent_variable]
#
#     df_dict = {}
#     for name in independent_variables:
#         df_dict[str(name)] = data[str(name)]
#
#     df_dict[str(dependent_variable)] = data[dependent_variable]
#     df_features_label = pd.DataFrame.from_dict(df_dict)
#
#     x = sm.add_constant(x)
#
#     if regularization:
#         model = sm.OLS(y,x).fit_regularized(method='elastic_net')
#     else:
#         #fig = plt.figure(1)
#         model = sm.OLS(y, x).fit()
#         fitted_value = model.fittedvalues
#         df_fitted_value = pd.DataFrame(fitted_value, columns=['fitted_values'])
#         # create instance of influence
#         influence = model.get_influence()
#
#         #sm.graphics.influence_plot(model)
#         #plt.show()
#
#         # obtain standardized residuals
#         standardized_residuals = influence.resid_studentized_internal
#         inf_sum = influence.summary_frame()
#
#         df_final_influence = pd.concat([df_features_label,inf_sum, df_fitted_value], axis=1).round(4)
#         print(df_final_influence)
#
#         student_resid = influence.resid_studentized_external
#         (cooks, p) = influence.cooks_distance
#         (dffits, p) = influence.dffits
#
#         df = model.summary()
#
#         results_as_html = df.tables[0].as_html()
#         df_0 = pd.read_html(results_as_html)[0].round(4)
#         df_new = df_0[[2, 3]]
#         df_0.drop(columns=[2, 3], inplace=True)
#         df_0 = pd.concat([df_0, df_new.rename(columns={2: 0, 3: 1})], ignore_index=True)
#         df_0.set_index(0, inplace=True)
#         df_0.index.name = None
#         df_0.rename(columns={1: 'Values'}, inplace = True)
#         df_0.drop(df_0.tail(2).index,inplace=True)
#
#         results_as_html = df.tables[1].as_html()
#         df_1 = pd.read_html(results_as_html)[0].round(4)
#         new_header = df_1.iloc[0, 1:]
#         df_1 = df_1[1:]
#         print(df_1.columns)
#         df_1.set_index(0, inplace=True)
#         df_1.columns = new_header
#         df_1.index.name = None
#
#         results_as_html = df.tables[2].as_html()
#         df_2 = pd.read_html(results_as_html)[0].round(4)
#         df_new = df_2[[2, 3]]
#         df_2.drop(columns=[2, 3], inplace=True)
#         df_2 = pd.concat([df_2, df_new.rename(columns={2: 0, 3: 1})], ignore_index=True)
#         df_2.set_index(0, inplace=True)
#         df_2.index.name = None
#         df_2.rename(columns={1: 'Values'}, inplace=True)
#
#     if not regularization:
#         white_test = het_white(model.resid, model.model.exog)
#         # define labels to use for output of White's test
#         labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']
#         results_dict = dict(zip(labels, white_test))
#         white_test = pd.DataFrame(results_dict.values(), index=results_dict.keys()).round(4)
#         white_test.rename(columns={0: 'Values'}, inplace=True)
#
#         bresuch_pagan_test = sms.het_breuschpagan(model.resid, model.model.exog)
#         # define labels to use for output of White's test
#         labels = ['Lagrange multiplier statistic', 'p-value','f-value', 'f p-value']
#         results_dict_bresuch = dict(zip(labels, bresuch_pagan_test))
#         bresuch_test = pd.DataFrame(results_dict_bresuch.values(), index=results_dict_bresuch.keys()).round(4)
#         bresuch_test.rename(columns={0: 'Values'}, inplace=True)
#
#         return {'DataFrame with all available influence results':df_final_influence.to_html(),'first_table': df_0.to_html(), 'second table': df_1.to_html(),
#                 'third table': df_2.to_html(), 'dataframe white test': white_test.to_html(), 'dataframe bresuch pagan test': bresuch_test.to_html()}
#     else:
#         return {'ll'}

@router.get("/linear_regressor_statsmodels")
async def linear_regression_statsmodels(step_id: str, run_id: str,
                                        dependent_variable: str,
                                        check_heteroscedasticity: bool | None = Query(default=True),
                                        regularization: bool | None = Query(default=False),
                                        independent_variables: list[str] | None = Query(default=None)):

    data = load_file_csv_direct(run_id, step_id)

    data = data.drop(['Unnamed: 0'], 1)

    x = data[independent_variables]
    y = data[dependent_variable]

    df_dict = {}
    for name in independent_variables:
        df_dict[str(name)] = data[str(name)]

    df_dict[str(dependent_variable)] = data[dependent_variable]
    df_features_label = pd.DataFrame.from_dict(df_dict)

    x = sm.add_constant(x)

    if regularization:
        model = sm.OLS(y,x).fit_regularized(method='elastic_net')
    else:
        #fig = plt.figure(1)
        model = sm.OLS(y, x).fit()
        # create instance of influence
        influence = model.get_influence()

        #sm.graphics.influence_plot(model)
        #plt.show()

        # obtain standardized residuals
        standardized_residuals = influence.resid_studentized_internal
        inf_sum = influence.summary_frame()

        df_final_influence = pd.concat([df_features_label,inf_sum], axis=1)
        inf_dict = {}
        for column in df_final_influence.columns:
            inf_dict[column] = list(df_final_influence[column])
        df_final_influence = df_final_influence.round(4)

        student_resid = influence.resid_studentized_external
        (cooks, p) = influence.cooks_distance
        (dffits, p) = influence.dffits

        df = model.summary()

        results_as_html = df.tables[0].as_html()
        df_0 = pd.read_html(results_as_html)[0]
        df_new = df_0[[2, 3]]
        df_0.drop(columns=[2, 3], inplace=True)
        df_0 = pd.concat([df_0, df_new.rename(columns={2: 0, 3: 1})], ignore_index=True)
        df_0.set_index(0, inplace=True)
        df_0.index.name = None
        df_0.rename(columns={1: 'Values'}, inplace = True)
        df_0.drop(df_0.tail(2).index,inplace=True)
        print(list(df_0.values))


        results_as_html = df.tables[1].as_html()
        df_1 = pd.read_html(results_as_html)[0]
        new_header = df_1.iloc[0, 1:]
        df_1 = df_1[1:]
        print(df_1.columns)
        df_1.set_index(0, inplace=True)
        df_1.columns = new_header
        df_1.index.name = None

        results_as_html = df.tables[2].as_html()
        df_2 = pd.read_html(results_as_html)[0]
        df_new = df_2[[2, 3]]
        df_2.drop(columns=[2, 3], inplace=True)
        df_2 = pd.concat([df_2, df_new.rename(columns={2: 0, 3: 1})], ignore_index=True)
        df_2.set_index(0, inplace=True)
        df_2.index.name = None
        df_2.rename(columns={1: 'Values'}, inplace=True)

    if not regularization:
        white_test = het_white(model.resid, model.model.exog)
        # define labels to use for output of White's test
        labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']
        results_dict = dict(zip(labels, white_test))
        white_test = pd.DataFrame(results_dict.values(), index=results_dict.keys())
        white_test.rename(columns={0: 'Values'}, inplace=True)

        bresuch_pagan_test = sms.het_breuschpagan(model.resid, model.model.exog)
        # define labels to use for output of White's test
        labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
        results_dict_bresuch = dict(zip(labels, bresuch_pagan_test))
        bresuch_test = pd.DataFrame(results_dict_bresuch.values(), index=results_dict_bresuch.keys())
        bresuch_test.rename(columns={0: 'Values'}, inplace=True)

        z = het_goldfeldquandt(y,x)
        labels = ['F-statistic', 'p-value', "ordering used in the alternative"]
        results_goldfeldquandt = dict(zip(labels, z))
        goldfeld_test = pd.DataFrame(results_goldfeldquandt.values(), index=results_goldfeldquandt.keys())
        goldfeld_test.rename(columns={0: 'Values'}, inplace=True)

        response = {'DataFrame with all available influence results':df_final_influence.to_html(),'first_table': df_0.to_json(orient='split'), 'second table': df_1.to_html(),
                'third table': df_2.to_dict(), 'dataframe white test': white_test.to_json(orient='split'),
                'dep': df_0.loc['Dep. Variable:'][0], 'model': df_0.loc['Model:'][0],
                'method': df_0.loc['Method:'][0], 'date': df_0.loc['Date:'][0],
                'time': df_0.loc['Time:'][0], 'no_obs': df_0.loc['No. Observations:'][0], 'resid': df_0.loc['Df Residuals:'][0],
                'df_model': df_0.loc['Df Model:'][0], 'cov_type': df_0.loc['Covariance Type:'][0],
                'r_squared': df_0.loc['R-squared:'][0], 'adj_r_squared': df_0.loc['Adj. R-squared:'][0],
                'f_stat': df_0.loc['F-statistic:'][0], 'prob_f': df_0.loc['Prob (F-statistic):'][0],
                'log_like': df_0.loc['Log-Likelihood:'][0], 'aic': df_0.loc['AIC:'][0], 'bic': df_0.loc['BIC:'][0],
                'omnibus': df_2.loc['Omnibus:'][0], 'prob_omni': df_2.loc['Prob(Omnibus):'][0], 'skew': df_2.loc['Skew:'][0], 'kurtosis': df_2.loc['Kurtosis:'][0],
                'durbin': df_2.loc['Durbin-Watson:'][0], 'jb': df_2.loc['Jarque-Bera (JB):'][0], 'prob_jb': df_2.loc['Prob(JB):'][0], 'cond': df_2.loc['Cond. No.'][0],
                'test_stat': white_test.loc['Test Statistic'][0], 'test_stat_p': white_test.loc['Test Statistic p-value'][0], 'white_f_stat': white_test.loc['F-Statistic'][0],
                'white_prob_f': white_test.loc['F-Test p-value'][0], 'influence_columns': list(df_final_influence.columns), 'influence_dict': inf_dict,
                'bresuch_lagrange': bresuch_test.loc['Lagrange multiplier statistic'][0], 'bresuch_p_value': bresuch_test.loc['p-value'][0],
                'bresuch_f_value': bresuch_test.loc['f-value'][0], 'bresuch_f_p_value': bresuch_test.loc['f p-value'][0],'Goldfeld-Quandt F-value':goldfeld_test.loc['F-statistic'][0],
                    'Goldfeld-Quandt p-value': goldfeld_test.loc['p-value'][0], 'Goldfeld-Quandt ordering used in the alternative': goldfeld_test.loc['ordering used in the alternative'][0]}
        return response
    else:
        return {'ll'}

@router.get("/transformation_methods")
async def transformation_methods(dependent_variable: str,
                                 method: str | None = Query("log",
                                                            regex="^(log)$|^(squared)$|^(root)$")):


    x = data[dependent_variable]

    if method == 'log':
        x = np.log(x)
    elif method == 'squared':
        x = np.sqrt(x)
    else:
        x = np.cbrt(x)

    return {'transformed array': x}

@router.get("/skewness_kurtosis")
async def skewness_kurtosis(dependent_variable: str):


    x = data[dependent_variable]

    skewness_res = skew(x)
    kurtosis_res = kurtosis(x)

    return {'skew': skewness_res, 'kurtosis': kurtosis_res}


@router.get("/z_score")
async def z_score(dependent_variable: str):


    x = data[dependent_variable]

    z_score_res = zscore(x)

    return {'z_score': list(z_score_res)}

@router.get("/logistic_regressor_statsmodels")
async def logistic_regression_statsmodels(dependent_variable: str,
                                          check_heteroscedasticity: bool | None = Query(default=True),
                                          regularization: bool | None = Query(default=True),
                                          independent_variables: list[str] | None = Query(default=None)):

    x = data[independent_variables]
    y = data[dependent_variable]

    x = sm.add_constant(x)

    if regularization:
        model = sm.Logit(y,x).fit_regularized(method='elastic_net')
    else:
        model = sm.Logit(y, x).fit()

        df = model.summary()

        results_as_html = df.tables[0].as_html()
        df_0 = pd.read_html(results_as_html)[0]

        results_as_html = df.tables[1].as_html()
        df_1 = pd.read_html(results_as_html)[0]

    if regularization==False:

        return {'first_table': df_0.to_json(orient="split"), 'second table': df_1.to_json(orient="split")}
    else:
        return {'ll'}

@router.get("/jarqueberatest")
async def jarqueberatest(dependent_variable: str):

    x = data[dependent_variable]

    jarque_bera_test = jarque_bera(x)

    statistic = jarque_bera_test.statistic
    pvalue = jarque_bera_test.pvalue

    if pvalue > 0.05:
        return {'statistic': statistic, 'pvalue': pvalue, 'Since this p-value is not less than .05, we fail to reject the null hypothesis. We don’t have sufficient evidence to say that this data has skewness and kurtosis that is significantly different from a normal distribution.':''}
    else:
        return {'statistic': statistic, 'pvalue': pvalue,'Since this p-value is less than .05, we reject the null hypothesis. Thus, we have sufficient evidence to say that this data has skewness and kurtosis that is significantly different from a normal distribution.':''}




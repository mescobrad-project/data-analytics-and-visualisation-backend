import random

import numpy as np
import pandas as pd
import json
from sklearn.cross_decomposition import CCA
from statsmodels.tsa.stattools import grangercausalitytests
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.utils import corr, cov
from factor_analyzer.confirmatory_factor_analyzer import ModelSpecification, ConfirmatoryFactorAnalyzer
from factor_analyzer import FactorAnalyzer
from scipy.stats import jarque_bera, fisher_exact, ranksums, chisquare, kruskal, alexandergovern, kendalltau, f_oneway, shapiro, \
    kstest, anderson, normaltest, boxcox, yeojohnson, bartlett, levene, fligner, obrientransform, pearsonr, spearmanr, \
    pointbiserialr, ttest_ind, mannwhitneyu, wilcoxon, ttest_rel, skew, kurtosis, probplot, zscore
from typing import Optional, Union, List
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.mediation import Mediation
import statsmodels.api as sm
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
import seaborn as sns
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, SGDClassifier, HuberRegressor,Lars, PoissonRegressor, LogisticRegression
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
from app.utils.utils_datalake import fget_object, get_saved_dataset_for_Hypothesis, upload_object
from app.utils.utils_general import get_local_storage_path, get_single_file_from_local_temp_storage, load_data_from_csv, \
    load_file_csv_direct
import scipy.stats as st
import statistics
from tabulate import tabulate
import seaborn as sns

from app.utils.utils_hypothesis import create_plots, compute_skewness, outliers_removal, compute_kurtosis

router = APIRouter()
data = pd.read_csv('example_data/mescobrad_dataset.csv')
data = data.drop(["Unnamed: 0"], axis=1)
# data = pd.read_csv('example_data/sample_questionnaire.csv')

def normality_test_content_results(column: str, selected_dataframe):
    if (column):
        # Creating Box-plot
        html_str_B = create_plots(plot_type='BoxPlot', column=column,second_column='', selected_dataframe=selected_dataframe)
        # Creating QQ-plot
        html_str = create_plots(plot_type='QQPlot', column=column, second_column='', selected_dataframe=selected_dataframe)
        # Creating Probability-plot
        html_str_P = create_plots(plot_type='PPlot', column=column, second_column='', selected_dataframe=selected_dataframe)
        #Creating histogram
        html_str_H = create_plots(plot_type='HistogramPlot', column=column, second_column='', selected_dataframe=selected_dataframe)
        skewtosend = compute_skewness(column, selected_dataframe)
        kurtosistosend = compute_kurtosis(column, selected_dataframe)
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

class FunctionOutputItem(BaseModel):
    """
    Known metadata information
    "files" : [["run_id: "string" , "step_id": "string"], "output":"string"]
     """
    run_id: str
    step_id: str
    file: str


@router.put("/save_hypothesis_output")
async def save_hypothesis_output(item: FunctionOutputItem) -> dict:
    output_json = json.loads(item.file)
    # print(output_json)
    try:
        path_to_storage = get_local_storage_path(item.run_id, item.step_id)
        out_filename = path_to_storage + '/output' + '/output.json'
        with open(out_filename, 'w') as fh:
            json.dump(output_json, fh, ensure_ascii=False)
        upload_object(bucket_name="demo", object_name='expertsystem/workflow/3fa85f64-5717-4562-b3fc-2c963f66afa6'
                                                     '/3fa85f64-5717-4562-b3fc-2c963f66afa6/3fa85f64-5717-4562-b3fc'
                                                     '-2c963f66afa6/Analytics_output.json', file=out_filename)
        # print(colorama.Fore.GREEN + "###################### Successfully! created json file. ##############################")
        print("###################### Successfully! created json file.")
        return '200'
    except Exception as e:
        print(e)
        print("Error : The save api")
        return '500'



@router.get("/return_columns")
async def name_columns(workflow_id: str, step_id: str, run_id: str):
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    data = load_data_from_csv(path_to_storage + "/" + name_of_file)

    # For the testing dataset
    if 'Unnamed: 0' in data.columns:
        data = data.drop(['Unnamed: 0'], axis=1)

    columns = data.columns
    return{'columns': list(columns), 'dataFrame': data.to_json(orient='records')}

@router.get("/return_binary_columns")
async def name_columns(workflow_id: str, step_id: str, run_id: str):
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    data = load_data_from_csv(path_to_storage + "/" + name_of_file)

    # For the testing dataset
    if 'Unnamed: 0' in data.columns:
        data = data.drop(['Unnamed: 0'], axis=1)
    for b_column in data.columns:
        if data[b_column].unique().shape[0] > 2:
            data = data.drop([b_column], axis=1)

    columns = data.columns
    return{'columns': list(columns)}


@router.get("/return_binary_columns")
async def name_columns(workflow_id: str, step_id: str, run_id: str):
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
    data = load_data_from_csv(path_to_storage + "/" + name_of_file)

    # For the testing dataset
    if 'Unnamed: 0' in data.columns:
        data = data.drop(['Unnamed: 0'], axis=1)
    for b_column in data.columns:
        if data[b_column].unique().shape[0] > 2:
            data = data.drop([b_column], axis=1)

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
async def normal_tests(workflow_id: str, step_id: str, run_id: str,
                       column: str,
                       nan_policy: Optional[str] | None = Query("propagate",
                                                                regex="^(propagate)$|^(raise)$|^(omit)$"),
                       axis: Optional[int] = 0,
                       alternative: Optional[str] | None = Query("two-sided",
                                                                 regex="^(two-sided)$|^(less)$|^(greater)$"),
                       name_test: str | None = Query("Shapiro-Wilk",
                                                   regex="^(Shapiro-Wilk)$|^(Kolmogorov-Smirnov)$|^(Anderson-Darling)$|^(D’Agostino’s K\^2)$|^(Jarque-Bera)$")) -> dict:

    data = load_file_csv_direct(workflow_id, run_id, step_id)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(['Unnamed: 0'], axis=1)
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
async def transform_data(workflow_id: str,
                         step_id: str,
                         run_id: str,
                         column: str,
                         name_transform: str | None = Query("Box-Cox",
                                                           regex="^(Box-Cox)$|^(Yeo-Johnson)$|^(Log)$|^(Squared-root)$|^(Cube-root)$"),
                         lmbd: Optional[float] = None,
                         alpha: Optional[float] = None) -> dict:

    data = load_file_csv_direct(workflow_id, run_id, step_id)
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


# @router.get("/compute_pearson_correlation", tags=['hypothesis_testing'])
# async def pearson_correlation(workflow_id: str, step_id: str, run_id: str, column_1: str, column_2: str):
#     data = load_file_csv_direct(workflow_id, run_id, step_id)
#     pearsonr_test = pearsonr(data[str(column_1)], data[str(column_2)])
#     return {'Pearson’s correlation coefficient':pearsonr_test[0], 'p-value': pearsonr_test[1]}
#
# @router.get("/compute_spearman_correlation", tags=['hypothesis_testing'])
# async def spearman_correlation(column_1: str, column_2: str):
#     spearman_test = spearmanr(data[str(column_1)], data[str(column_2)])
#     return {'Spearman correlation coefficient': spearman_test[0], 'p-value': spearman_test[1]}
#
# @router.get("/compute_kendalltau_correlation", tags=['hypothesis_testing'])
# async def kendalltau_correlation(column_1: str,
#                                  column_2: str,
#                                  nan_policy: Optional[str] | None = Query("propagate",
#                                                                            regex="^(propagate)$|^(raise)$|^(omit)$"),
#                                  alternative: Optional[str] | None = Query("two-sided",
#                                                                            regex="^(two-sided)$|^(less)$|^(greater)$"),
#                                  variant: Optional[str] | None = Query("b",
#                                                                        regex="^(b)$|^(c)$"),
#                                  method: Optional[str] | None = Query("auto",
#                                                                       regex="^(auto)$|^(asymptotic)$|^(exact)$")):
#     kendalltau_test = kendalltau(data[str(column_1)], data[str(column_2)], nan_policy=nan_policy, alternative=alternative, variant=variant, method=method)
#     return {'kendalltau correlation coefficient': kendalltau_test[0], 'p-value': kendalltau_test[1]}

@router.get("/compute_point_biserial_correlation", tags=['hypothesis_testing'])
async def point_biserial_correlation(workflow_id: str, step_id: str, run_id: str,
                                     column_1: str, column_2: str,
                                     # remove_outliers: bool | None = Query(default=True)
                                     ):
    data = load_file_csv_direct(workflow_id, run_id, step_id)
    unique_values = np.unique(data[str(column_1)])
    unique_values.sort()
    if len(unique_values) == 2:
        html_scr = create_plots(plot_type='Scatter_Two_Variables', column=column_1, second_column=column_2, selected_dataframe=data)
        sub_set_a = data[data[str(column_1)] != unique_values[1]]
        sub_set_b = data[data[str(column_1)] != unique_values[0]]
        new_dataset_for_bp = [sub_set_a[str(column_2)], sub_set_b[str(column_2)]]
        html_box = create_plots(plot_type='BoxPlot', column=column_2, second_column=column_1, selected_dataframe=new_dataset_for_bp)
        html_hist_A = create_plots(plot_type='HistogramPlot', column=column_2, second_column='',
                                   selected_dataframe=sub_set_a)
        html_hist_B = create_plots(plot_type='HistogramPlot', column=column_2, second_column='',
                                   selected_dataframe=sub_set_b)
        # find outliers
        sub_set_a_clean, outliers_a = outliers_removal(column_2, sub_set_a)
        sub_set_b_clean, outliers_b = outliers_removal(column_2, sub_set_b)
        # check Normality per sample
        shapiro_test_A = shapiro(sub_set_a[str(column_2)])
        shapiro_test_B = shapiro(sub_set_b[str(column_2)])
        # check homoscedasticity per sample
        Levene_A = levene(sub_set_a[str(column_2)], sub_set_b[str(column_2)], center='median')
        # Levene_B = ''
        df = sub_set_a_clean.append(sub_set_b_clean)
        pointbiserialr_test = pointbiserialr(df[str(column_1)], df[str(column_2)])
        return {'status': 'OK',
                'error_descr': '',
                'scatter_plot': html_scr,
                'html_box': html_box,
                'sample_A': {
                    'value': str(unique_values[0]),
                    'N': len(sub_set_a),
                    'N_clean':  len(sub_set_a_clean),
                    'outliers': outliers_a[column_2].to_json(orient='records'),
                    'html_hist': html_hist_A,
                    'Norm_statistic': shapiro_test_A.statistic,
                    'Norm_p_value': shapiro_test_A.pvalue,
                    'Hom_statistic': Levene_A.statistic,
                    'Hom_p_value': Levene_A.pvalue
                },
                'sample_B': {
                    'value': str(unique_values[1]),
                    'N': len(sub_set_b),
                    'N_clean': len(sub_set_b_clean),
                    'outliers': outliers_b[column_2].to_json(orient='records'),
                    'html_hist': html_hist_B,
                    'Norm_statistic': shapiro_test_B.statistic,
                    'Norm_p_value': shapiro_test_B.pvalue,
                    # 'Hom_statistic': Levene_B.statistic,
                    # 'Hom_p_value': Levene_B.pvalue
                },
                'correlation': pointbiserialr_test[0],
                'p_value': pointbiserialr_test[1],
                'new_dataset': df.to_json(orient='records')
                }
    else:
        return {'status': 'Error',
                'error_descr': 'The selected variable is not dichotomous.',
                'scatter_plot': '',
                'html_box': '',
                'sample_A': {
                    'value': '',
                    'N': '',
                    'N_clean':  '',
                    'outliers': '',
                    'html_hist': '',
                    'Norm_statistic': '',
                    'Norm_p_value': '',
                    'Hom_statistic': '',
                    'Hom_p_value': '',
                },
                'sample_B': {
                    'value': '',
                    'N': '',
                    'N_clean': '',
                    'outliers': '',
                    'html_hist': '',
                    'Norm_statistic': '',
                    'Norm_p_value': '',
                    'Hom_statistic': '',
                    'Hom_p_value': '',
                },
                'correlation': '',
                'p_value': '',
                'new_dataset': []}


@router.get("/check_homoscedasticity", tags=['hypothesis_testing'])
async def check_homoskedasticity(workflow_id: str,
                                 step_id: str,
                                 run_id: str,
                                 columns: list[str] | None = Query(default=None),
                                 name_of_test: str | None = Query("Levene",
                                                                  regex="^(Levene)$|^(Bartlett)$|^(Fligner-Killeen)$"),
                                 center: Optional[str] | None = Query("median",
                                                                      regex="^(trimmed)$|^(median)$|^(mean)$")):
    data = load_file_csv_direct(workflow_id, run_id, step_id)

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
    print(*args)
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
async def statistical_tests(workflow_id: str,
                            step_id: str,
                            run_id: str,
                            column_1: str,
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
    data = load_file_csv_direct(workflow_id, run_id, step_id)

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
async def LDA(workflow_id: str,
                step_id: str,
                run_id: str,
              dependent_variable: str,
              solver: str | None = Query("svd",
                                         regex="^(svd)$|^(lsqr)$|^(eigen)$"),
              shrinkage_1: str | None = Query("none",
                                              regex="^(none)$|^(auto)$|^(float)$"),
              shrinkage_2: float | None = Query(default=None, gt=-1, lt=1),
              # shrinkage_3: float | None = Query(default=None),
              independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    # dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    features_columns = dataset.columns
    X = np.array(dataset)
    Y = np.array(df_label.astype('float64'))

    # target_names = np.unique(Y)
    # sc = StandardScaler()
    # X = sc.fit_transform(X)
    # # print(X)
    # le = LabelEncoder()
    # Y = le.fit_transform(Y)
    # # print(Y)
    # # X_train = clf.fit_transform(X, Y)
    # # # print('X_train:')
    # # # print(X_train)
    # # # # colors = ["navy", "turquoise", "darkorange"]
    # # # # for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    # # colors = ["navy", "darkorange"] + ["#"+''.join([random.choice('0123456789ABCDEF')
    # #                                                for j in range(6)]) for i in range(number_of_classes-2)]
    # # fig = plt.figure(figsize=(14, 9))
    # # ax = fig.add_subplot(111,
    # #                      projection='3d')
    # #
    # # for color, i, target_name in zip(colors, classes, target_names):
    # #     ax.scatter(
    # #         X_train[Y == i, 0], Y, alpha=0.8, color=color, label=target_name
    # #         # X_train[Y == i, 0], X_train[Y == i, 1], X_train[Y == i, 2], alpha=0.8, color=color, label=target_name
    # #     )
    # # plt.legend(loc="best", shadow=False, scatterpoints=1)
    # # plt.title("LDA of IRIS dataset")
    # # plt.show()

    if solver == 'lsqr' or solver == 'eigen':
        if shrinkage_1 == 'float':
            clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage_2)
        elif shrinkage_1 == 'auto':
            clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage_1)
        else:
            clf = LinearDiscriminantAnalysis(solver=solver)
    else:
        clf = LinearDiscriminantAnalysis(solver=solver)
    # print(solver)
    clf.fit(X,Y)

    classes = clf.classes_
    number_of_classes = len(clf.classes_)
    number_of_components = min(len(clf.classes_) - 1, clf.n_features_in_)
    if solver == 'svd':
        df_xbar = pd.DataFrame(clf.xbar_, columns=['xbar'])
        df_xbar.insert(loc=0, column='Feature', value=features_columns)
        df_scalings = pd.DataFrame(clf.scalings_, columns=[i + 1 for i in range(number_of_components)])
        df_scalings.insert(loc=0, column='Feature', value=features_columns)
    else:
        df_xbar = pd.DataFrame()
        df_scalings = pd.DataFrame()

    df_mean = pd.DataFrame(clf.means_, columns=features_columns)
    df_mean.insert(loc=0, column='Class', value=classes)
    df_prior = pd.DataFrame(clf.priors_, columns=['priors'])
    df_prior.insert(loc=0, column='Class', value=classes)

    if solver == 'eigen' or solver =='svd':
        df_explained_variance_ratio = pd.DataFrame(clf.explained_variance_ratio_, columns=['Variance ratio'])
        df_explained_variance_ratio.insert(loc=0, column='Component', value=[i + 1 for i in range(number_of_components)])
    else:
        df_explained_variance_ratio = pd.DataFrame()

    df_coefs = pd.DataFrame(clf.coef_, columns=features_columns)
    df_intercept = pd.DataFrame(clf.intercept_, columns=['intercept'])
    df_coefs['intercept'] = df_intercept['intercept']
    if df_coefs.shape[0] == len(classes):
        df_coefs.insert(loc=0, column='Class', value=classes)
    try:
        to_return = {
            'number_of_features': int(clf.n_features_in_),
            'features_columns': features_columns.tolist(),
            'number_of_classes':number_of_classes,
            'classes_': clf.classes_.tolist(),
            'number_of_components': number_of_components,
            'explained_variance_ratio': df_explained_variance_ratio.to_json(orient='records'),
            'means_': df_mean.to_json(orient='records'),
            'priors_': df_prior.to_json(orient='records'),
            'scalings_': df_scalings.to_json(orient='records'),
            'xbar_': df_xbar.to_json(orient='records'),
            'coefficients': df_coefs.to_json(orient='records'),
            'intercept': df_intercept.to_json(orient='records')
        }
        print(to_return)
        return to_return
    except Exception as e:
        print(e)
        print("Error : Creating QQPlot")
        return {}

    # return {'coefficients': df_coefs.to_json(orient='split'), 'intercept': df_intercept.to_json(orient='split')}


@router.get("/principal_component_analysis")
async def principal_component_analysis(workflow_id: str,
                                       step_id: str,
                                       run_id: str,
                                       n_components_1: int | None = Query(default=None),
                                       n_components_2: float | None = Query(default=None, gt=0, lt=1),
                                       independent_variables: list[str] | None = Query(default=None)):
    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
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
        pca_t = pca.fit_transform(X)
        principal_Df = pd.DataFrame(data=pca_t, columns=["principalcomponent1", "principalcomponent2"])
        # principal_Df = pd.DataFrame(data=pca_t,
        #                             columns=['principal component 1', 'principal component 2'])
        print(principal_Df)
        print(dataset)
        pca.fit(X)
    else:
        pca = PCA(n_components=n_components_2)
        pca.fit(X)

    # principal_Df = pd.DataFrame(data=pca, columns=['principal component 1', 'principal component 2'])

    return {
        'columns': dataset.columns.tolist(),
        'n_features_': pca.n_features_,
            'n_features_in_': pca.n_features_in_,
            'n_samples_': pca.n_samples_,
            'random_state': pca.random_state,
            'iterated_power': pca.iterated_power,
            'mean_': pca.mean_.tolist(),
            'explained_variance_': pca.explained_variance_.tolist(),
            'noise_variance_': pca.noise_variance_,
            'pve': pca.explained_variance_ratio_.tolist(),
            'singular_values': pca.singular_values_.tolist(),
            'principal_axes': pca.components_[0].tolist()}
    # return {'Percentage of variance explained by each of the selected components': pca.explained_variance_ratio_.tolist(),
    #             'The singular values corresponding to each of the selected components. ': pca.singular_values_.tolist(),
    #             'Principal axes in feature space, representing the directions of maximum variance in the data.' : pca.components_.tolist()}

@router.get("/kmeans_clustering")
async def kmeans_clustering(workflow_id: str,
                            step_id: str,
                            run_id: str,
                            n_clusters: int,
                            independent_variables: list[str] | None = Query(default=None)):
    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    return {'Coordinates of cluster centers': kmeans.cluster_centers_.tolist(),
            'Labels of each point ': kmeans.labels_.tolist(),
            'Sum of squared distances of samples to their closest cluster center' : kmeans.inertia_}

# TODO DELETE NEWER IMPLEMENTATION LATER IN THE FILE
# @router.get("/linear_regressor")
# async def linear_regression(dependent_variable: str,
#                             independent_variables: list[str] | None = Query(default=None)):
#     dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
#     df_label = dataset[dependent_variable]
#     for columns in dataset.columns:
#         if columns not in independent_variables:
#             dataset = dataset.drop(str(columns), axis=1)
#
#     X = np.array(dataset)
#     Y = np.array(df_label)
#
#     clf = LinearRegression()
#
#     clf.fit(X, Y)
#     if np.shape(X)[1] == 1:
#         coeffs = clf.coef_
#         inter = clf.intercept_
#         df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
#         df_names = pd.DataFrame(dataset.columns, columns=['variables'])
#         df = pd.concat([df_names, df_coeffs], axis=1)
#         return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_json(orient='split')}
#     else:
#         coeffs = np.squeeze(clf.coef_)
#         inter = clf.intercept_
#         df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
#         df_names = pd.DataFrame(dataset.columns, columns=['variables'])
#         df = pd.concat([df_names, df_coeffs], axis=1)
#         return {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
#                 'dataframe': df.to_json(orient='split')}

# TODO Create frontend
@router.get("/elastic_net")
async def elastic_net(workflow_id: str,
                      step_id: str,
                      run_id: str,
                      dependent_variable: str,
                      alpha: float | None = Query(default=1.0),
                      l1_ratio: float | None = Query(default=0.5, ge=0, le=1),
                      max_iter: int | None = Query(default=1000),
                      independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)


    X = np.array(dataset)
    Y = np.array(df_label.astype('float64'))


    clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)

    clf.fit(X, Y)

    residuals = Y - clf.predict(X)
    skew_res = skew(residuals)
    kurt_res = kurtosis(residuals)
    jarq_res = jarque_bera(residuals)
    stat_jarq = jarq_res.statistic
    p_jarq = jarq_res.pvalue
    omn_res_stat, omn_res_p = normaltest(residuals)
    durb_res = durbin_watson(residuals)

    df_for_scatter = pd.DataFrame(data={'Actual Values': list(Y), 'Predicted Values': list(clf.predict(X)),
                                        'Residuals': list(Y - clf.predict(X))})
    values_dict = {}
    for column in df_for_scatter.columns:
        values_dict[column] = list(df_for_scatter[column])


    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'skew': skew_res,
                'kurtosis': kurt_res,
                'Jarque Bera statistic':stat_jarq,
                'Jarque-Bera p-value': p_jarq,
                'Omnibus test statistic': omn_res_stat,
                'Omnibus test p-value': omn_res_p,
                'Durbin Watson': durb_res,
                'actual_values': list(Y),
                'predicted values': list(clf.predict(X)),
                'residuals': list(Y-clf.predict(X)),
                'coefficient of determination (R^2)':clf.score(X,Y),
                'coefficients': coeffs.tolist(),
                'intercept': inter.tolist(), 'dataframe': df.to_html(), 'values_dict': values_dict,
                'values_columns': list(df_for_scatter.columns),
                'values_df': df_for_scatter.to_html()}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'skew': skew_res,
                'kurtosis': kurt_res,
                'Jarque Bera statistic':stat_jarq,
                'Jarque-Bera p-value': p_jarq,
                'Omnibus test statistic': omn_res_stat,
                'Omnibus test p-value': omn_res_p,
                'Durbin Watson': durb_res,
                'actual_values': list(Y),
                'predicted values': list(clf.predict(X)),
                'residuals': list(Y-clf.predict(X)),
                'coefficient of determination (R^2)':clf.score(X,Y),
                'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_html(), 'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
                'values_df': df_for_scatter.to_html()}

# TODO Create frontend
@router.get("/lasso_regression")
async def lasso(workflow_id: str,
                step_id: str,
                run_id: str,
                dependent_variable: str,
                alpha: float | None = Query(default=1.0, gt=0),
                max_iter: int | None = Query(default=1000),
                independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label.astype('float64'))

    clf = Lasso(alpha=alpha, max_iter=max_iter)

    clf.fit(X, Y)
    residuals = Y - clf.predict(X)
    skew_res = skew(residuals)
    kurt_res = kurtosis(residuals)
    jarq_res = jarque_bera(residuals)
    stat_jarq = jarq_res.statistic
    p_jarq = jarq_res.pvalue
    omn_res_stat, omn_res_p = normaltest(residuals)
    durb_res = durbin_watson(residuals)

    df_for_scatter = pd.DataFrame(data={'Actual Values': list(Y), 'Predicted Values': list(clf.predict(X)),
                                        'Residuals': list(Y - clf.predict(X))})
    values_dict = {}
    for column in df_for_scatter.columns:
        values_dict[column] = list(df_for_scatter[column])

    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'skew': skew_res,
                'kurtosis': kurt_res,
                'Jarque Bera statistic':stat_jarq,
                'Jarque-Bera p-value': p_jarq,
                'Omnibus test statistic': omn_res_stat,
                'Omnibus test p-value': omn_res_p,
                'Durbin Watson': durb_res,
                'actual_values': list(Y),
                'predicted values': list(clf.predict(X)),
                'residuals': list(Y-clf.predict(X)),
                'coefficient of determination (R^2)':clf.score(X,Y),
                'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_html(),
                'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
                'values_df': df_for_scatter.to_html()}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'skew': skew_res,
                'kurtosis': kurt_res,
                'Jarque Bera statistic':stat_jarq,
                'Jarque-Bera p-value': p_jarq,
                'Omnibus test statistic': omn_res_stat,
                'Omnibus test p-value': omn_res_p,
                'Durbin Watson': durb_res,
                'actual_values': list(Y),
                'predicted values': list(clf.predict(X)),
                'residuals': list(Y-clf.predict(X)),
                'coefficient of determination (R^2)':clf.score(X,Y),
                'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_html(), 'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
                'values_df': df_for_scatter.to_html()}

# TODO Create frontend
@router.get("/ridge_regression")
async def ridge(workflow_id: str,
                step_id: str,
                run_id: str,
                dependent_variable: str,
                alpha: float | None = Query(default=1.0, gt=0),
                max_iter: int | None = Query(default=None),
                solver: str | None = Query("auto",
                                           regex="^(auto)$|^(svd)$|^(cholesky)$|^(sparse_cg)$|^(lsqr)$|^(sag)$|^(lbfgs)$"),
                independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label.astype('float64'))

    if solver!='lbfgs':
        clf = Ridge(alpha=alpha, max_iter=max_iter, solver=solver)
    else:
        clf = Ridge(alpha=alpha, max_iter=max_iter, solver=solver, positive=True)

    clf.fit(X, Y)
    residuals = Y - clf.predict(X)
    skew_res = skew(residuals)
    kurt_res = kurtosis(residuals)
    jarq_res = jarque_bera(residuals)
    stat_jarq = jarq_res.statistic
    p_jarq = jarq_res.pvalue
    omn_res_stat, omn_res_p = normaltest(residuals)
    durb_res = durbin_watson(residuals)

    df_for_scatter = pd.DataFrame(data={'Actual Values': list(Y), 'Predicted Values': list(clf.predict(X)),
                                        'Residuals': list(Y - clf.predict(X))})
    values_dict = {}
    for column in df_for_scatter.columns:
        values_dict[column] = list(df_for_scatter[column])

    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'skew': skew_res,
                'kurtosis': kurt_res,
                'Jarque Bera statistic':stat_jarq,
                'Jarque-Bera p-value': p_jarq,
                'Omnibus test statistic': omn_res_stat,
                'Omnibus test p-value': omn_res_p,
                'Durbin Watson': durb_res,
                'actual_values': list(Y),
                'predicted values': list(clf.predict(X)),
                'residuals': list(Y-clf.predict(X)),
                'coefficient of determination (R^2)':clf.score(X,Y),
                'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_html(),
                'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
                'values_df': df_for_scatter.to_html()}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'skew': skew_res,
                'kurtosis': kurt_res,
                'Jarque Bera statistic':stat_jarq,
                'Jarque-Bera p-value': p_jarq,
                'Omnibus test statistic': omn_res_stat,
                'Omnibus test p-value': omn_res_p,
                'Durbin Watson': durb_res,
                'actual_values': list(Y),
                'predicted values': list(clf.predict(X)),
                'residuals': list(Y-clf.predict(X)),
                'coefficient of determination (R^2)':clf.score(X,Y),
                'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_html(), 'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
                'values_df': df_for_scatter.to_html()}

def full_log_likelihood(w, X, y):
    score = np.dot(X, w).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def null_log_likelihood(w, X, y):
    z = np.array([w if i == 0 else 0.0 for i, w in enumerate(w.reshape(1, X.shape[1])[0])]).reshape(X.shape[1], 1)
    score = np.dot(X, z).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def mcfadden_rsquare(w, X, y):
    return 1.0 - (full_log_likelihood(w, X, y) / null_log_likelihood(w, X, y))


@router.get("/logistic_regression_sklearn")
async def sklearn_logistic_regression(workflow_id: str,
                                      step_id: str,
                                      run_id: str,
                                      dependent_variable: str,
                                      C: float | None = Query(default=1.0),
                                      l1_ratio: float | None = Query(default=None, gt=0, le=1),
                                      max_iter: int | None = Query(default=100),
                                      penalty: str | None = Query("l2",
                                                                 regex="^(l2)$|^(l1)$|^(elasticnet)$|^(None)$"),
                                      solver: str | None = Query("lbfgs",
                                                                 regex="^(lbfgs)$|^(liblinear)$|^(newton-cg)$|^(newton-cholesky)$|^(sag)$|^(saga)$"),
                                      independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    dataset_names = dataset.columns
    X = np.array(dataset)
    Y = np.array(df_label.astype('float64'))

    if solver == 'lbfgs':
        if penalty == 'l2' or penalty == 'None':
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C)
        else:
            return {'This combination is not supported'}
    elif solver == 'liblinear':
        if penalty == 'l1' or penalty == 'l2':
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C)
        else:
            return {'This combination is not supported'}
    elif solver == 'newton-cg':
        if penalty == 'l2' or penalty == 'None':
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C)
        else:
            return {'This combination is not supported'}
    elif solver == 'newton-cholesky':
        if penalty == 'l2' or penalty == 'None':
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C)
        else:
            return {'This combination is not supported'}
    elif solver == 'sag':
        if penalty == 'l2' or penalty == 'None':
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C)
        else:
            return {'This combination is not supported'}
    else:
        if penalty == 'elasticnet':
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C, l1_ratio=l1_ratio)
        else:
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C)

    clf.fit(X, Y)

    coeffs = clf.coef_
    inter = clf.intercept_
    df_coeffs = pd.DataFrame(coeffs, columns=dataset_names)

    w = np.array(coeffs).transpose()

    return {'Log-Likelihood (full)':full_log_likelihood(w, X, Y),
            'Log-Likelihood (Null - model with only intercept)': null_log_likelihood(w, X, Y),
            'Pseudo R-squar. (McFadden’s R^2)': mcfadden_rsquare(w, X, Y),
            'intercept': inter.tolist(), 'dataframe': df_coeffs.to_html()}



def full_log_likelihood(w, X, y):
    score = np.dot(X, w).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def null_log_likelihood(w, X, y):
    z = np.array([w if i == 0 else 0.0 for i, w in enumerate(w.reshape(1, X.shape[1])[0])]).reshape(X.shape[1], 1)
    score = np.dot(X, z).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def mcfadden_rsquare(w, X, y):
    return 1.0 - (full_log_likelihood(w, X, y) / null_log_likelihood(w, X, y))


@router.get("/logistic_regression_sklearn")
async def sklearn_logistic_regression(workflow_id: str,
                                      step_id: str,
                                      run_id: str,
                                      dependent_variable: str,
                                      C: float | None = Query(default=1.0),
                                      l1_ratio: float | None = Query(default=None, gt=0, le=1),
                                      max_iter: int | None = Query(default=100),
                                      penalty: str | None = Query("l2",
                                                                 regex="^(l2)$|^(l1)$|^(elasticnet)$|^(None)$"),
                                      solver: str | None = Query("lbfgs",
                                                                 regex="^(lbfgs)$|^(liblinear)$|^(newton-cg)$|^(newton-cholesky)$|^(sag)$|^(saga)$"),
                                      independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    dataset_names = dataset.columns
    X = np.array(dataset)
    Y = np.array(df_label.astype('float64'))

    if solver == 'lbfgs':
        if penalty == 'l2' or penalty == 'None':
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C)
        else:
            return {'This combination is not supported'}
    elif solver == 'liblinear':
        if penalty == 'l1' or penalty == 'l2':
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C)
        else:
            return {'This combination is not supported'}
    elif solver == 'newton-cg':
        if penalty == 'l2' or penalty == 'None':
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C)
        else:
            return {'This combination is not supported'}
    elif solver == 'newton-cholesky':
        if penalty == 'l2' or penalty == 'None':
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C)
        else:
            return {'This combination is not supported'}
    elif solver == 'sag':
        if penalty == 'l2' or penalty == 'None':
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C)
        else:
            return {'This combination is not supported'}
    else:
        if penalty == 'elasticnet':
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C, l1_ratio=l1_ratio)
        else:
            clf = LogisticRegression(penalty=penalty, max_iter=max_iter, solver=solver, C=C)

    clf.fit(X, Y)

    coeffs = clf.coef_
    inter = clf.intercept_
    df_coeffs = pd.DataFrame(coeffs, columns=dataset_names)

    w = np.array(coeffs).transpose()

    return {'Log-Likelihood (full)':full_log_likelihood(w, X, Y),
            'Log-Likelihood (Null - model with only intercept)': null_log_likelihood(w, X, Y),
            'Pseudo R-squar. (McFadden’s R^2)': mcfadden_rsquare(w, X, Y),
            'intercept': inter.tolist(), 'dataframe': df_coeffs.to_json(orient='split')}



@router.get("/sgd_regression")
async def sgd_regressor(workflow_id: str,
                        step_id: str,
                        run_id: str,
                        dependent_variable: str,
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

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label.astype('float64'))

    if loss == 'huber' or loss == 'epsilon_insensitive' or loss == 'squared_epsilon_insensitive':
        if learning_rate == 'constant' or learning_rate == 'invscaling' or learning_rate == 'adaptive':
            clf = SGDRegressor(alpha=alpha, max_iter=max_iter, epsilon=epsilon, eta0=eta0, penalty=penalty, l1_ratio=l1_ratio, learning_rate=learning_rate)
        else:
            clf = SGDRegressor(alpha=alpha, max_iter=max_iter, epsilon=epsilon, penalty=penalty, l1_ratio=l1_ratio, learning_rate=learning_rate)
    else:
        clf = SGDRegressor(alpha=alpha, max_iter=max_iter, eta0=eta0, penalty=penalty, l1_ratio=l1_ratio, learning_rate=learning_rate)

    clf.fit(X, Y)

    residuals = Y - clf.predict(X)
    skew_res = skew(residuals)
    kurt_res = kurtosis(residuals)
    jarq_res = jarque_bera(residuals)
    stat_jarq = jarq_res.statistic
    p_jarq = jarq_res.pvalue
    omn_res_stat, omn_res_p = normaltest(residuals)
    durb_res = durbin_watson(residuals)

    df_for_scatter = pd.DataFrame(data={'Actual Values': list(Y), 'Predicted Values': list(clf.predict(X)),
                                        'Residuals': list(Y - clf.predict(X))})
    values_dict = {}
    for column in df_for_scatter.columns:
        values_dict[column] = list(df_for_scatter[column])

    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'skew': skew_res,
                'kurtosis': kurt_res,
                'Jarque Bera statistic':stat_jarq,
                'Jarque-Bera p-value': p_jarq,
                'Omnibus test statistic': omn_res_stat,
                'Omnibus test p-value': omn_res_p,
                'Durbin Watson': durb_res,
                'actual_values': list(Y),
                'predicted values': list(clf.predict(X)),
                'residuals': list(Y-clf.predict(X)),
                'coefficient of determination (R^2)':clf.score(X,Y),
                'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_html(),
                'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
                'values_df': df_for_scatter.to_html()}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'skew': skew_res,
                'kurtosis': kurt_res,
                'Jarque Bera statistic':stat_jarq,
                'Jarque-Bera p-value': p_jarq,
                'Omnibus test statistic': omn_res_stat,
                'Omnibus test p-value': omn_res_p,
                'Durbin Watson': durb_res,
                'actual_values': list(Y),
                'predicted values': list(clf.predict(X)),
                'residuals': list(Y-clf.predict(X)),
                'coefficient of determination (R^2)':clf.score(X,Y),
                'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_html(), 'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
                'values_df': df_for_scatter.to_html()}

@router.get("/huber_regression")
async def huber_regressor(workflow_id: str, step_id: str, run_id: str,
                          dependent_variable: str,
                          max_iter: int | None = Query(default=1000),
                          epsilon: float | None = Query(default=1.5, gt=1),
                          alpha: float | None = Query(default=0.0001,ge=0),
                          independent_variables: list[str] | None = Query(default=None)):

    # dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label.astype('float64'))

    clf = HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=max_iter)

    clf.fit(X, Y)

    residuals = Y - clf.predict(X)
    skew_res = skew(residuals)
    kurt_res = kurtosis(residuals)
    jarq_res = jarque_bera(residuals)
    stat_jarq = jarq_res.statistic
    p_jarq = jarq_res.pvalue
    omn_res_stat, omn_res_p = normaltest(residuals)
    durb_res = durbin_watson(residuals)

    df_for_scatter = pd.DataFrame(data={'Actual Values': list(Y), 'Predicted Values': list(clf.predict(X)),
                                        'Residuals': list(Y - clf.predict(X))})
    values_dict = {}
    for column in df_for_scatter.columns:
        values_dict[column] = list(df_for_scatter[column])

    if np.shape(X)[1] == 1:
        coeffs = clf.coef_
        inter = clf.intercept_
        outliers = clf.outliers_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'skew': skew_res,
                'kurtosis': kurt_res,
                'Jarque Bera statistic':stat_jarq,
                'Jarque-Bera p-value': p_jarq,
                'Omnibus test statistic': omn_res_stat,
                'Omnibus test p-value': omn_res_p,
                'Durbin Watson': durb_res,
                'actual_values': list(Y),
                'predicted values': list(clf.predict(X)),
                'residuals': list(Y-clf.predict(X)),
                'coefficient of determination (R^2)':clf.score(X,Y),
                'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'outliers':outliers.tolist(), 'dataframe': df.to_html(),
                'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
                'values_df': df_for_scatter.to_html()}
    else:
        coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        outliers = clf.outliers_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        return {'skew': skew_res,
                'kurtosis': kurt_res,
                'Jarque Bera statistic':stat_jarq,
                'Jarque-Bera p-value': p_jarq,
                'Omnibus test statistic': omn_res_stat,
                'Omnibus test p-value': omn_res_p,
                'Durbin Watson': durb_res,
                'actual_values': list(Y),
                'predicted values': list(clf.predict(X)),
                'residuals': list(Y-clf.predict(X)),
                'coefficient of determination (R^2)':clf.score(X,Y),
                'coefficients': coeffs.tolist(), 'outliers':outliers.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_html(), 'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
                'values_df': df_for_scatter.to_html()}

@router.get("/linearsvr_regression")
async def linear_svr_regressor(workflow_id: str,
                               step_id: str,
                               run_id: str,
                               dependent_variable: str,
                               max_iter: int | None = Query(default=1000),
                               epsilon: float | None = Query(default=0),
                               C: float | None = Query(default=1,gt=0),
                               loss: str | None = Query("epsilon_insensitive",
                                                         regex="^(epsilon_insensitive)$|^(squared_epsilon_insensitive)$"),
                               independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label.astype('float64'))

    clf = LinearSVR(loss=loss, C=C, epsilon=epsilon, max_iter=max_iter)

    clf.fit(X, Y)
    residuals = Y-clf.predict(X)
    skew_res = skew(residuals)
    kurt_res = kurtosis(residuals)
    jarq_res = jarque_bera(residuals)
    stat_jarq = jarq_res.statistic
    p_jarq = jarq_res.pvalue
    print(p_jarq)
    omn_res_stat, omn_res_p = normaltest(residuals)
    durb_res = durbin_watson(residuals)

    df_for_scatter = pd.concat([pd.DataFrame(data={'Actual Values': list(Y), 'Predicted Values': list(clf.predict(X)),
                                        'Residuals': list(Y - clf.predict(X))}), dataset], axis=1)
    values_dict = {}
    for column in df_for_scatter.columns:
        values_dict[column] = list(df_for_scatter[column])

    coeffs = clf.coef_
    inter = clf.intercept_
    df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
    df_names = pd.DataFrame(dataset.columns, columns=['variables'])
    df = pd.concat([df_names, df_coeffs], axis=1)
    return {'skew': skew_res,
            'kurtosis': kurt_res,
            'Jarque Bera statistic':stat_jarq,
            'Jarque-Bera p-value': p_jarq,
            'Omnibus test statistic': omn_res_stat,
            'Omnibus test p-value': omn_res_p,
            'Durbin Watson': durb_res,
            'actual_values': list(Y),
            'predicted values': list(clf.predict(X)),
            'residuals': list(Y-clf.predict(X)),
            'coefficient of determination (R^2)':clf.score(X,Y),
            'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_json(orient='records'),
            'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
            'values_df': df_for_scatter.to_json(orient='records')}



@router.get("/linearsvc_regression")
async def linear_svc_regressor(workflow_id: str,
                               step_id: str,
                               run_id: str,
                               dependent_variable: str,
                               max_iter: int | None = Query(default=1000),
                               C: float | None = Query(default=1,gt=0),
                               loss: str | None = Query("hinge",
                                                         regex="^(hinge)$|^(squared_hinge)$"),
                               penalty: str | None = Query("l2",
                                                         regex="^(l1)$|^(l2)$"),
                               independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)


    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    features_columns = dataset.columns

    X = np.array(dataset)
    Y = np.array(df_label.astype('float64'))

    if loss == 'hinge' and penalty == 'l1':
        return {'This combination is not supported.'}
    else:
        clf = LinearSVC(loss=loss, C=C, penalty=penalty, max_iter=max_iter)

    clf.fit(X, Y)

    df_coefs = pd.DataFrame(clf.coef_, columns=features_columns)

    df_intercept = pd.DataFrame(clf.intercept_, columns=['intercept'])

    df_for_scatter = pd.concat([df_coefs, df_intercept], axis=1)

    # df_for_scatter = df_for_scatter.fillna('')

    values_dict = {}
    for column in df_for_scatter.columns:
        values_dict[column] = list(df_for_scatter[column])

    return {'coefficients': df_coefs.to_html(), 'intercept': df_intercept.to_html(), 'values_dict': values_dict,
            'values_columns': list(df_for_scatter.columns),
            'values_df': df_for_scatter.to_html(), 'dataframe': df_for_scatter.to_html()}

@router.get("/ancova")
async def ancova_2(workflow_id: str,
                    step_id: str,
                    run_id: str,
                   dv: str,
                   between: str,
                   covar: list[str] | None = Query(default=None),
                   effsize: str | None = Query("np2",
                                               regex="^(np2)$|^(n2)$")):

    # df_data = pd.read_csv('example_data/mescobrad_dataset.csv')
    df_data = load_file_csv_direct(workflow_id, run_id, step_id)

    df = ancova(data=df_data, dv=dv, covar=covar, between=between, effsize=effsize)
    df = df.fillna('')
    all_res = []
    for ind, row in df.iterrows():
        temp_to_append = {
            'id': ind,
            'Source': row['Source'],
            'SS': row['SS'],
            'DF': row['DF'],
            'F': row['F'],
            'p-unc': row['p-unc'],
            'np2': row['np2']
        }
        all_res.append(temp_to_append)
    return {'DataFrame': all_res}
    # return {'ANCOVA':df.to_json(orient="split")}

@router.get("/linear_mixed_effects_model")
async def linear_mixed_effects_model(workflow_id: str,
                    step_id: str,
                    run_id: str,
                     dependent: str,
                     groups: str,
                     independent: list[str] | None = Query(default=None),
                     use_sqrt: bool | None = Query(default=True)):

    # data = pd.read_csv('example_data/mescobrad_dataset.csv')
    data = load_file_csv_direct(workflow_id, run_id, step_id)
    z = dependent + "~"
    for i in range(len(independent)):
        z = z + "+" + independent[i]

    md = smf.mixedlm(z, data, groups=data[groups], use_sqrt=use_sqrt)
    mdf = md.fit()
    df = mdf.summary()
    df_0 = df.tables[0]
    tbl1_res = []
    for ind, row in df_0.iterrows():
        temp_to_append = {
            'id': ind,
            "col0": row[0],
            "col1": row[1],
            "col2": row[2],
            "col3": row[3],
        }
        tbl1_res.append(temp_to_append)
    df_1 = df.tables[1]
    tbl2_res = []
    for ind, row in df_1.iterrows():
        temp_to_append = {
            'id': ind,
            "col0": row[0],
            "col1": row[1],
            "col2": row[2],
            "col3": row[3],
            "col4": row[4],
            "col5": row[5],
        }
        tbl2_res.append(temp_to_append)
    print(df)

    return {'first_table': tbl1_res, 'second_table': tbl2_res}
    # return {'first_table': df_0.to_json(orient='split'), 'second_table': df_1.to_json(orient='split')}

@router.get("/poisson_regression")
async def poisson_regression(workflow_id: str,
                             step_id: str,
                             run_id: str,
                             dependent_variable: str,
                             alpha: float | None = Query(default=1.0, ge=0),
                             max_iter: int | None = Query(default=1000),
                             independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    df_label = dataset[dependent_variable]
    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)
    Y = np.array(df_label.astype('float64'))

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
async def cox_regression(workflow_id: str,
                         step_id: str,
                         run_id: str,
                         duration_col: str,
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

    to_return = []

    fig = plt.figure(1)
    ax = plt.subplot(111)
    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    # dataset = pd.read_csv('example_data/mescobrad_dataset.csv')

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
    tbl1_res = []
    for ind, row in df.iterrows():
        temp_to_append = {
            'id': ind,
            "col0": row[0],
            "col1": row[1],
            "col2": row[2],
            "col3": row[3],
            "col4": row[4],
            "col5": row[5],
            "col6": row[6],
            "col7": row[7],
            "col8": row[8],
            "col9": row[9],
            "col10": row[10],
        }
        tbl1_res.append(temp_to_append)
    #fig = plt.figure(figsize=(18, 12))
    cph.plot(hazard_ratios=hazard_ratios, ax=ax)
    plt.show()

    html_str = mpld3.fig_to_html(fig)
    to_return.append({"figure_1": html_str})
    #plt.close(1)
    #fig = plt.figure(2)
    #ax = plt.subplot(121)
    plt.clf()
    if values!=None:
        cph.plot_partial_effects_on_outcome(covariates=covariates, values=values, cmap='coolwarm')
        plt.show()
        html_str = mpld3.fig_to_html(fig)
        to_return.append({"figure_2": html_str})
        # to_return["figure_2"] = html_str

    results = proportional_hazard_test(cph, dataset, time_transform='rank')

    df_1 = results.summary
    tbl2_res = []
    for ind, row in df_1.iterrows():
        temp_to_append = {
            'id': ind,
            "test_statistic": row[0],
            "p": row[1],
            "-log2(p)": row[2]
        }
        tbl2_res.append(temp_to_append)

    AIC = cph.AIC_partial_
    return {'Concordance_Index':cph.concordance_index_,'AIC': AIC, 'Dataframe': tbl1_res, 'figure': to_return, 'proportional_hazard_test': tbl2_res}
    # return {'Concordance Index':cph.concordance_index_ ,'Akaike information criterion (AIC) (partial log-likelihood)': AIC,'Dataframe of the coefficients, p-values, CIs, etc.':df.to_json(orient="split"), 'figure': to_return, 'proportional hazard test': df_1.to_json(orient='split')}

@router.get("/time_varying_covariates")
async def time_varying_covariates(
        workflow_id: str,
         step_id: str,
         run_id: str,
          event_col: str,
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

    to_return = []

    fig = plt.figure(1)
    ax = plt.subplot(111)
    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    # dataset = pd.read_csv('example_data/mescobrad_dataset.csv')

    dataset_long = to_episodic_format(dataset, duration_col=duration_col, event_col=event_col, time_gaps=time_gaps)

    if correction_columns:
        dataset_long[column_1+'*'+column_2] = dataset_long[column_1]*dataset_long[column_2]

    cph = CoxTimeVaryingFitter(alpha=alpha, penalizer=penalizer, l1_ratio=l1_ratio)

    cph.fit(dataset_long, event_col=event_col, id_col='id', weights_col=weights_col,start_col='start', stop_col='stop',strata=strata)

    df = cph.summary
    tbl1_res = []
    for ind, row in df.iterrows():
        temp_to_append = {
            'id': ind,
            "col0": row[0],
            "col1": row[1],
            "col2": row[2],
            "col3": row[3],
            "col4": row[4],
            "col5": row[5],
            "col6": row[6],
            "col7": row[7],
            "col8": row[8],
            "col9": row[9],
            "col10": row[10],
        }
        tbl1_res.append(temp_to_append)
    #fig = plt.figure(figsize=(18, 12))
    cph.plot(ax=ax)
    plt.show()

    html_str = mpld3.fig_to_html(fig)
    to_return.append({"figure_1": html_str})

    return {'AIC':cph.AIC_partial_,'Dataframe':tbl1_res, 'figure': to_return}
    # return {'Akaike information criterion (AIC) (partial log-likelihood)':cph.AIC_partial_,'Dataframe of the coefficients, p-values, CIs, etc.':df.to_json(orient="split"), 'figure': to_return}

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
async def generalized_estimating_equations(workflow_id: str,
                                           step_id: str,
                                           run_id: str,
                                           dependent_variable: str,
                                           groups: str,
                                           independent_variables: list[str] | None = Query(default=None),
                                           cov_struct: str | None = Query("independence",
                                                                           regex="^(independence)$|^(autoregressive)$|^(exchangeable)$|^(nested_working_dependence)$"),
                                           family: str | None = Query("poisson",
                                                                      regex="^(poisson)$|^(gamma)$|^(gaussian)$|^(inverse_gaussian)$|^(negative_binomial)$|^(binomial)$|^(tweedie)$")):

    data = load_file_csv_direct(workflow_id, run_id, step_id)

    z = dependent_variable + "~"
    for i in range(len(independent_variables)):
        z = z + "+" + independent_variables[i]

    print(family)

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

    if cov_struct == "independence":
        ind = sm.cov_struct.Independence()
    elif cov_struct == "autoregressive":
        ind = sm.cov_struct.Autoregressive()
    elif cov_struct == "exchangeable":
        ind = sm.cov_struct.Exchangeable()
    else:
        ind = sm.cov_struct.Nested()

    print(cov_struct)
    print(fam)
    print(ind)


    md = smf.gee(z, groups, data, cov_struct=ind, family=fam)

    mdf = md.fit()

    df = mdf.summary()

    # print(df)

    results_as_html = df.tables[0].as_html()
    df_0 = pd.read_html(results_as_html)[0]
    df_new = df_0[[2, 3]]
    df_0.drop(columns=[2, 3], inplace=True)
    df_0 = pd.concat([df_0, df_new.rename(columns={2: 0, 3: 1})], ignore_index=True)
    df_0.set_index(0, inplace=True)
    df_0.index.name = None
    df_0.rename(columns={1: 'Values'}, inplace=True)
    df_0.drop(df_0.tail(2).index, inplace=True)
    df_0.reset_index(inplace=True)
    # print(list(df_0.values))

    results_as_html = df.tables[1].as_html()
    df_1 = pd.read_html(results_as_html)[0]
    new_header = df_1.iloc[0, 1:]
    df_1 = df_1[1:]
    df_1.set_index(0, inplace=True)
    df_1.columns = new_header
    df_1.index.name = None
    df_1.reset_index(inplace=True)
    df_1.rename(columns={'[0.025': '0.025', '0.975]': '0.975'}, inplace=True)

    results_as_html = df.tables[2].as_html()
    df_2 = pd.read_html(results_as_html)[0]
    df_new = df_2[[2, 3]]
    df_2.drop(columns=[2, 3], inplace=True)
    df_2 = pd.concat([df_2, df_new.rename(columns={2: 0, 3: 1})], ignore_index=True)
    df_2.set_index(0, inplace=True)
    df_2.index.name = None
    df_2.rename(columns={1: 'Values'}, inplace=True)
    df_2.reset_index(inplace=True)

    # print(df_1)
    #
    # print(df_0)
    # # print(df_0['No. Observations: '])
    print(df)

    return {'first_table':df_0.to_json(orient='records'),
            'second_table':df_1.to_json(orient='records'),
            'third_table':df_2.to_json(orient='records')}

@router.get("/kaplan_meier")
async def kaplan_meier(workflow_id: str,
                       step_id: str,
                       run_id: str,
                       column_1: str,
                       column_2: str,
                       at_risk_counts: bool | None = Query(default=True),
                       label: str | None = Query(default=None),
                       alpha: float | None = Query(default=0.05)):
    # to_return = {}
    #
    # fig = plt.figure(1)
    # ax = plt.subplot(111)

    # dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    kmf = KaplanMeierFitter(alpha=alpha, label=label)
    kmf.fit(dataset[column_1], dataset[column_2])
    kmf.plot_survival_function(at_risk_counts=at_risk_counts)
    plt.ylabel("Survival probability")
    plt.savefig(path_to_storage + "/output/survival_function.svg", format="svg")
    plt.show()


    df = kmf.survival_function_
    timeline = pd.DataFrame(kmf.timeline)
    conditional_time_to_event = pd.DataFrame(kmf.conditional_time_to_event_)
    confidence_interval = kmf.confidence_interval_
    event_table = kmf.event_table
    confidence_interval_cumulative_density = kmf.confidence_interval_cumulative_density_
    cumulative_density = kmf.cumulative_density_
    median_survival_time = kmf.median_survival_time_

    df.insert(0, "timeline", timeline)
    confidence_interval.insert(0, "timeline", timeline)
    confidence_interval.columns = confidence_interval.columns.str.replace('.', ',', regex=True)
    conditional_time_to_event.insert(0, "timeline", timeline)
    event_table.insert(0, "event_at", timeline)
    confidence_interval_cumulative_density.insert(0, "timeline", timeline)
    confidence_interval_cumulative_density.columns = confidence_interval_cumulative_density.columns.str.replace('.', ',', regex=True)
    cumulative_density.insert(0, "timeline", timeline)


    return {"survival_function":df.to_json(orient="records"),
            "confidence_interval": confidence_interval.to_json(orient='records'),
            'event_table': event_table.to_json(orient="records"),
            "conditional_time_to_event": conditional_time_to_event.to_json(orient="records"),
            "confidence_interval_cumulative_density":confidence_interval_cumulative_density.to_json(orient="records"),
            "cumulative_density" : cumulative_density.to_json(orient='records'),
            "timeline" : timeline.to_json(orient='records'),
            "median_survival_time": str(median_survival_time)}


@router.get("/fisher")
async def fisher(
        workflow_id: str,
        step_id: str,
        run_id: str,
        variable_column: str,
        variable_row: str,
        # variable_bottom_left: int,
        # variable_bottom_right: int,
        alternative: Optional[str] | None = Query("two-sided",
                                                  regex="^(two-sided)$|^(less)$|^(greater)$")):

    data = load_file_csv_direct(workflow_id, run_id, step_id)
    row_var = data[variable_row]
    column_var = data[variable_column]
    # df = [[variable_top_left,variable_top_right], [variable_bottom_left,variable_bottom_right]]

    df = pd.crosstab(index=row_var,columns=column_var)
    df1 = pd.crosstab(index=row_var,columns=column_var, margins=True, margins_name= "Total")

    odd_ratio, p_value = fisher_exact(df, alternative=alternative)

    return {'odd_ratio': odd_ratio, "p_value": p_value, "crosstab":df1.to_json(orient='split')}

@router.get("/mc_nemar")
async def mc_nemar(workflow_id: str,
                   step_id: str,
                   run_id: str,
                   variable_column: str,
                   variable_row: str,
                   exact: bool | None = Query(default=False),
                   correction: bool | None = Query(default=True)):

    # df = [[variable_top_left,variable_top_right], [variable_bottom_left,variable_bottom_right]]
    data = load_file_csv_direct(workflow_id, run_id, step_id)
    row_var = data[variable_row]
    column_var = data[variable_column]
    df = pd.crosstab(index=row_var,columns=column_var)
    df1 = pd.crosstab(index=row_var,columns=column_var, margins=True, margins_name= "Total")

    result = mcnemar(df, exact=exact, correction=correction)

    return {'statistic': result.statistic, "p_value": result.pvalue, "crosstab":df1.to_json(orient='split')}

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
async def risk_ratio_1(
        workflow_id: str,
        step_id: str,
        run_id: str,
        exposure: str,
        outcome: str,
        time: str | None = Query(default=None),
        reference: int | None = Query(default=0),
        alpha: float | None = Query(default=0.05),
        method: str | None = Query("risk_ratio",
                                   regex="^(risk_ratio)$|^(risk_difference)$|^(number_needed_to_treat)$|^(odds_ratio)$|^(incidence_rate_ratio)$|^(incidence_rate_difference)$")):

    to_return = {}

    fig = plt.figure(1)
    ax = plt.subplot(111)

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    # zepid.datasets
    # dataset = load_sample_data(False)
    # print(load_sample_data(False))
    if method == 'risk_ratio':
        rr = RiskRatio(reference=reference, alpha=alpha)
        rr.fit(dataset, exposure=exposure, outcome=outcome)
    elif method == 'risk_difference':
        rr = RiskDifference(reference=reference, alpha=alpha)
        rr.fit(dataset, exposure=exposure, outcome=outcome)
    elif method == 'number_needed_to_treat':
        rr = NNT(reference=reference, alpha=alpha)
        rr.fit(dataset, exposure=exposure, outcome=outcome)
        df = rr.results
        return {'table': df.to_json(orient="records")}
    elif method == 'odds_ratio':
        rr = OddsRatio(reference=reference, alpha=alpha)
        rr.fit(dataset, exposure=exposure, outcome=outcome)
    elif method == 'incidence_rate_ratio':
        rr = IncidenceRateRatio(reference=reference, alpha=alpha)
        rr.fit(dataset, exposure=exposure, outcome=outcome, time=time)
    elif method == "incidence_rate_difference":
        rr = IncidenceRateDifference(reference=reference, alpha=alpha)
        rr.fit(dataset, exposure=exposure, outcome=outcome, time=time)
    else:
        return {'table':''}

    df = rr.results
    rr.plot()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    plt.savefig(path_to_storage +"/output/Risktest.svg", format="svg")
    return {'table': df.to_json(orient="records")}

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
async def risk_ratio_function(workflow_id: str,
                              step_id: str,
                              run_id: str,
                              exposed_with: int,
                              unexposed_with: int,
                              exposed_without: int,
                              unexposed_without: int,
                              alpha: float | None = Query(default=0.05)):

    r = risk_ratio(a=exposed_with, b=unexposed_with, c=exposed_without, d=unexposed_without, alpha=alpha)
    print(r)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error
    return {'estimated_risk': estimated_risk, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'standard_error': standard_error}

@router.get("/risk_difference_function")
async def risk_difference_function(
        workflow_id: str,
        step_id: str,
        run_id: str,
        exposed_with: int,
        unexposed_with: int,
        exposed_without: int,
        unexposed_without: int,
        alpha: float | None = Query(default=0.05)):

    r = risk_difference(a=exposed_with, b=unexposed_with, c=exposed_without, d=unexposed_without, alpha=alpha)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error

    return {'risk_difference': estimated_risk, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'standard_error': standard_error}

@router.get("/number_needed_to_treat_function")
async def number_needed_to_treat_function(
        workflow_id: str,
        step_id: str,
        run_id: str,
        exposed_with: int,
        unexposed_with: int,
        exposed_without: int,
        unexposed_without: int,
        alpha: float | None = Query(default=0.05)):

    r = number_needed_to_treat(a=exposed_with, b=unexposed_with, c=exposed_without, d=unexposed_without, alpha=alpha)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error

    return {'nnt': estimated_risk, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'standard_error': standard_error}

@router.get("/odds_ratio_function")
async def odds_ratio_function(
        workflow_id: str,
        step_id: str,
        run_id: str,
        exposed_with: int,
        unexposed_with: int,
        exposed_without: int,
        unexposed_without: int,
        alpha: float | None = Query(default=0.05)):

    r = odds_ratio(a=exposed_with, b=unexposed_with, c=exposed_without, d=unexposed_without, alpha=alpha)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error

    return {'odds_ratio': estimated_risk, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'standard_error': standard_error}

@router.get("/incidence_rate_ratio_function")
async def incidence_rate_ratio_function(
        workflow_id: str,
        step_id: str,
        run_id: str,
        exposed_with: int,
        unexposed_with: int,
        person_time_exposed: int,
        person_time_unexposed: int,
        alpha: float | None = Query(default=0.05)):

    r = incidence_rate_ratio(a=exposed_with, c=unexposed_with, t1=person_time_exposed, t2=person_time_unexposed, alpha=alpha)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error
    print(r)
    return {'incident_rate_ratio': estimated_risk, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'standard_error': standard_error}

@router.get("/incidence_rate_difference_function")
async def incidence_rate_difference_function(
        workflow_id: str,
        step_id: str,
        run_id: str,
        exposed_with: int,
        unexposed_with: int,
        person_time_exposed: int,
        person_time_unexposed: int,
        alpha: float | None = Query(default=0.05)):

    r = incidence_rate_difference(a=exposed_with, c=unexposed_with, t1=person_time_exposed, t2=person_time_unexposed, alpha=alpha)
    estimated_risk = r.point_estimate
    lower_bound = r.lower_bound
    upper_bound = r.upper_bound
    standard_error = r.standard_error

    return {'incident_rate_difference': estimated_risk, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'standard_error': standard_error}

@router.get("/correlations_pingouin")
async def correlations_pingouin(workflow_id: str,
                                step_id: str,
                                run_id: str,
                                # column_1: str,
                                column_2: list[str] | None = Query(default=None),
                                alternative: Optional[str] | None = Query("two-sided",
                                                                          regex="^(two-sided)$|^(less)$|^(greater)$"),
                                method: Optional[str] | None = Query("pearson",
                                                                     regex="^(pearson)$|^(spearman)$|^(kendall)$|^(bicor)$|^(percbend)$|^(shepherd)$|^(skipped)$")):
    data = load_file_csv_direct(workflow_id, run_id, step_id)
    df = data[column_2]

    df1 = df.rcorr(stars=False).round(5)
    corrs = df.corr()

    # mask = np.zeros_like(corrs)
    # mask[np.triu_indices_from(mask)] = True
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # # ax = sns.heatmap(corrs, annot=True, cmap='Spectral_r', mask=mask, square=True, vmin=-1, vmax=1)
    # # plt.xticks(range(len(corrs)), corrs.columns)
    # # print(range(len(corrs)), corrs.columns)
    #
    # ax = plt.matshow(df.corr())
    # plt.title('Correlation matrix')
    # plt.show()
    # ss = mpld3.save_json(fig, 'ss.json')
    # html_str = mpld3.fig_to_html(fig)

    all_res = []
    count=0
    for i in column_2:
        for j in column_2:
            if i == j or column_2.index(j) < column_2.index(i):
                continue
            res = pingouin.corr(x=data[i], y=data[j], method=method, alternative=alternative).round(5)
            res.insert(0,'Cor', i + "-" + j, True)
            count = count + 1
            for ind, row in res.iterrows():
                temp_to_append = {
                    "id": count,
                    "Cor": row['Cor'],
                    "n": row['n'],
                    "r": row['r'],
                    "CI95%": "[" + str(row['CI95%'].item(0)) + "," + str(row['CI95%'].item(1)) + "]",
                    "p-val": row['p-val'],
                    "power": row['power']
                }
                if method == 'pearson':
                    temp_to_append["BF10"] = row['BF10']
                if method == 'shepherd':
                    temp_to_append["outliers"] = row['outliers']
            all_res.append(temp_to_append)

    return {'DataFrame': all_res, "Table_rcorr": df1.to_json(orient='records')}
    # return {'DataFrame': all_res, "Table_rcorr": df1.to_json(orient='records'), "rplot":html_str}

@router.get("/linear_regressor_pinguin")
async def linear_regression_pinguin(dependent_variable: str,
                                    alpha: float | None=Query(default=0.05),
                                    relimp: bool | None=Query(default=False),
                                    independent_variables: list[str] | None = Query(default=None)):

    lm = pingouin.linear_regression(data[independent_variables], data[dependent_variable], as_dataframe=True, alpha=alpha, relimp=relimp)

    return {'residuals': lm.residuals_.tolist(), 'degrees of freedom of the model': lm.df_model_, 'degrees of freedom of the residuals': lm.df_resid_ , 'dataframe': lm.to_json(orient='split')}

@router.get("/logistic_regressor_pinguin")
async def logistic_regression_pinguin(workflow_id: str, step_id: str, run_id: str,
                                      dependent_variable: str,
                                      alpha: float | None=Query(default=0.05),
                                      independent_variables: list[str] | None = Query(default=None)):

    data = load_file_csv_direct(workflow_id, run_id, step_id)

    lm = pingouin.logistic_regression(data[independent_variables], data[dependent_variable], as_dataframe=True, alpha=alpha)
    print(lm.columns)
    values_dict = {}
    for column in lm.columns:
        values_dict[column] = list(lm[column])

    return {'dataframe': lm.to_html(), 'values_dict': values_dict, 'values_columns': list(lm.columns)}

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
async def linear_regression_statsmodels(workflow_id: str, step_id: str, run_id: str,
                                        dependent_variable: str,
                                        check_heteroscedasticity: bool | None = Query(default=True),
                                        regularization: bool | None = Query(default=False),
                                        independent_variables: list[str] | None = Query(default=None)):
    data = load_file_csv_direct(workflow_id, run_id, step_id)

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
        fitted_value = model.fittedvalues
        df_fitted_value = pd.DataFrame(fitted_value, columns=['fitted_values'])
        resid_value = model.resid
        df_resid_value = pd.DataFrame(resid_value, columns=['residuals'])
        # create instance of influence
        influence = model.get_influence()

        #sm.graphics.influence_plot(model)
        #plt.show()

        # obtain standardized residuals
        standardized_residuals = influence.resid_studentized_internal
        inf_sum = influence.summary_frame()

        df_final_influence = pd.concat([df_features_label,inf_sum,df_fitted_value,df_resid_value], axis=1)
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
        # print(list(df_0.values))

        results_as_html = df.tables[1].as_html()
        df_1 = pd.read_html(results_as_html)[0]
        new_header = df_1.iloc[0, 1:]
        df_1 = df_1[1:]
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
async def logistic_regression_statsmodels(workflow_id: str, step_id: str, run_id: str,
                                          dependent_variable: str,
                                          independent_variables: list[str] | None = Query(default=None)):

    data = load_file_csv_direct(workflow_id, run_id, step_id)

    x = data[independent_variables]
    y = data[dependent_variable]

    x = sm.add_constant(x)

    model = sm.Logit(y, x).fit()

    df = model.summary()



    results_as_html = df.tables[0].as_html()
    df_0 = pd.read_html(results_as_html)[0]
    df_new = df_0[[2, 3]]
    df_0.drop(columns=[2, 3], inplace=True)
    df_0 = pd.concat([df_0, df_new.rename(columns={2: 0, 3: 1})], ignore_index=True)
    df_0.set_index(0, inplace=True)
    df_0.index.name = None
    df_0.rename(columns={1: 'Values'}, inplace=True)

    results_as_html = df.tables[1].as_html()
    df_1 = pd.read_html(results_as_html)[0]
    new_header = df_1.iloc[0, 1:]
    df_1 = df_1[1:]
    df_1.set_index(0, inplace=True)
    df_1.columns = new_header
    df_1.index.name = None



    df_1.fillna('', inplace=True)
    values_dict = {}
    for column in df_1.columns:
        values_dict[column] = list(df_1[column])

    print(df_0.index)

    return {'first_table': df_0.to_html(), 'second table': df_1.to_html(),
            'values_dict': values_dict, 'values_columns': list(df_1.columns),
            'dep': df_0.loc['Dep. Variable:'][0], 'model': df_0.loc['Model:'][0],
            'method': df_0.loc['Method:'][0], 'date': df_0.loc['Date:'][0],
            'time': df_0.loc['Time:'][0], 'no_obs': df_0.loc['No. Observations:'][0],
            'resid': df_0.loc['Df Residuals:'][0],
            'df_model': df_0.loc['Df Model:'][0], 'cov_type': df_0.loc['Covariance Type:'][0],
            'pseudo_r_squared': df_0.loc['Pseudo R-squ.:'][0], 'log_like': df_0.loc['Log-Likelihood:'][0],
            'LL-Null': df_0.loc['LL-Null:'][0], 'LLR p-value': df_0.loc['LLR p-value:'][0],
            'converged': df_0.loc['converged:'][0]}

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


@router.get("/correlation_matrix")
async def correlation(workflow_id: str,
                      step_id: str,
                      run_id: str,
                      independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    r = corr(dataset)

    return {'Correlation Matrix': r.tolist()}


@router.get("/covariance_matrix")
async def covariance(workflow_id: str,
                     step_id: str,
                     run_id: str,
                     ddof : int | None = Query(default=0),
                     independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    r = cov(dataset, ddof=ddof)

    return {'Covariance Matrix': r.tolist()}

@router.get("/choose_number_of_factors")
async def choose_number_of_factors(workflow_id: str,
                                   step_id: str,
                                   run_id: str,
                                   use_smc : bool | None = Query(default=True),
                                   n_factors: int | None = Query(default=3),
                                   rotation : str | None = Query("None",
                                                                 regex="^(None)$|^(varimax)$|^(promax)$|^(oblimin)$|^(oblimax)$|^(quartimin)$|^(quartimax)$|^(equamax)$|^(geomin_obl)$|^(geomin_ort)$"),
                                   method: str | None = Query("minres",
                                                              regex="^(minres)$|^(ml)$|^(principal)$"),
                                   impute: str | None = Query("drop",
                                                              regex="^(drop)$|^(mean)$|^(median)$"),
                                   independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)


    if rotation == str(None):
        fa = FactorAnalyzer(n_factors=n_factors, rotation=None, method=method, use_smc=use_smc, impute=impute)
    else:
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method, use_smc=use_smc, impute=impute)


    fa.fit(dataset)

    original_eigen_values, common_factor_eigen_values = fa.get_eigenvalues()

    to_return = {}

    fig = plt.figure(1)

    plt.scatter(range(1, dataset.shape[1] + 1), original_eigen_values)
    plt.plot(range(1, dataset.shape[1] + 1), original_eigen_values)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

    html_str = mpld3.fig_to_html(fig)
    to_return["figure_1"] = html_str

    return {'Original Eigenvalues': original_eigen_values.tolist(),
            'Figure': to_return}


@router.get("/calculate_factor_analysis")
async def compute_factor_analysis(workflow_id: str,
                                  step_id: str,
                                  run_id: str,
                                  use_smc : bool | None = Query(default=True),
                                  n_factors: int | None = Query(default=3),
                                  rotation : str | None = Query("None",
                                                                regex="^(None)$|^(varimax)$|^(promax)$|^(oblimin)$|^(oblimax)$|^(quartimin)$|^(quartimax)$|^(equamax)$|^(geomin_obl)$|^(geomin_ort)$"),
                                  method: str | None = Query("minres",
                                                             regex="^(minres)$|^(ml)$|^(principal)$"),
                                  impute: str | None = Query("drop",
                                                             regex="^(drop)$|^(mean)$|^(median)$"),
                                  independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)


    if rotation == str(None):
        fa = FactorAnalyzer(n_factors=n_factors, rotation=None, method=method, use_smc=use_smc, impute=impute)
    else:
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method, use_smc=use_smc, impute=impute)


    fa.fit(dataset)

    original_eigen_values, common_factor_eigen_values = fa.get_eigenvalues()
    factor_variance, proportional_factor_variance, cumulative_variance = fa.get_factor_variance()
    uniquenesses = fa.get_uniquenesses()

    new_dataset = fa.transform(dataset)
    print(fa.phi_)

    df = pd.DataFrame(data=dataset.values, columns=independent_variables)
    corrs = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corrs, cmap='Spectral_r', square=True, annot=True)
    plt.title('Correlation matrix')
    plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + '/output/correlation_matrix.png')
    plt.show()
    correlation_matrix = mpld3.fig_to_html(fig)

    factor_list = ['Factor'+str(i+1) for i in range(n_factors)]

    df_factor_loadings = pd.DataFrame(data=fa.loadings_, index=independent_variables, columns=factor_list)
    print(df_factor_loadings)
    df_factor_loadings = df_factor_loadings.reset_index().rename(columns={'index': 'Variables'})
    df_corr = pd.DataFrame(data=fa.corr_, index=independent_variables, columns=independent_variables)
    fig, ax = plt.subplots()
    ax.matshow(df_corr, cmap='viridis')
    plt.xticks(ticks=range(len(df_corr)), labels=independent_variables)
    plt.yticks(ticks=range(len(df_corr)), labels=independent_variables)
    for (i, j), z in np.ndenumerate(df_corr.values):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    plt.show()


    df_com_eigen = pd.DataFrame(data={'Variables': independent_variables, 'Communalities': fa.get_communalities(), 'Original Eigenvalues': original_eigen_values,
                                'Common Factor Eigenvalues': common_factor_eigen_values, 'Uniquenesses from the factor loading matrix': uniquenesses})
    print(len(new_dataset))
    df_factor_variances = pd.DataFrame(data={'Factors': factor_list, 'Factor variances': factor_variance, 'Proportional Factor Variance': proportional_factor_variance,
                                     'Cumulative Factor Variance': cumulative_variance})
    df_new_dataset = pd.DataFrame(data=new_dataset, columns=factor_list)





    # df_factor_correlations = pd.DataFrame(data=fa.phi_, columns=factor_list).corr()
    # fig, ax = plt.subplots()
    # sns.heatmap(df_factor_correlations, cmap='Spectral_r', square=True, annot=True)
    # plt.title('Correlation matrix')
    # plt.show()
    # correlation_matrix = mpld3.fig_to_html(fig)

    to_return = {'factor_matrix': df_factor_loadings.to_json(orient='records'),
                 'corr_matrix': correlation_matrix,
                 'df_com_eigen': df_com_eigen.to_json(orient='records'),
                 'df_factor_variances': df_factor_variances.to_json(orient='records'),
                 'df_new_dataset': df_new_dataset.to_json(orient='records'),
                 'rotation': str(rotation),
                 'df_structure': None,
                 'df_rotation': None,
                 'factor_corr_matrix': None}

    if rotation == str(None):
        # return{'factor_matrix_test': fa.loadings_.tolist(), #Factor Loadings matrix
        #     'factor_matrix': df_factor_loadings.to_json(orient='records'),
        #        'corr_matrix': correlation_matrix,
        #        'orig_corr_matrix': fa.corr_.tolist(), #Original Correlation Matrix
        #        'df_com_eigen': df_com_eigen.to_json(orient='records'),
        #        'Communalities': fa.get_communalities().tolist(),
        #        'Original Eigenvalues': original_eigen_values.tolist(),
        #        'Common Factor Eigenvalues': common_factor_eigen_values.tolist(),
        #        'df_factor_variances': df_factor_variances.to_json(orient='records'),
        #        'Factor variances': factor_variance.tolist(),
        #        'Proportional Factor Variance': proportional_factor_variance.tolist(),
        #        'Cumulative Factor Variance': cumulative_variance.tolist(),
        #        'uniquenesses from the factor loading matrix': uniquenesses.tolist(),
        #        'df_new_dataset': df_new_dataset.to_json(orient='records'),
        #        'Factor scores for a new dataset': new_dataset.tolist()}
        pass
    else:
        df_rotation = pd.concat([pd.DataFrame(data=factor_list, columns=['Factors']),
                                 pd.DataFrame(data=fa.rotation_matrix_, columns=factor_list)],
                                 axis=1)
        to_return['df_rotation'] = df_rotation.to_json(orient='records')
        if rotation in ["promax", "oblimin", "quartimin"]:
            df_factor_corr_matrix = pd.DataFrame(data=fa.phi_, columns=factor_list, index=factor_list)
            fig, ax = plt.subplots()
            sns.heatmap(df_factor_corr_matrix, cmap='Spectral_r', square=True, annot=True)
            plt.title('Factor correlation matrix')
            plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + '/output/factor_correlation_matrix.png')
            plt.show()
            factor_corr_matrix = mpld3.fig_to_html(fig)
            to_return['factor_corr_matrix'] = factor_corr_matrix
            if rotation == 'promax':
                df_structure = pd.concat([pd.DataFrame(data=independent_variables, columns=['Variables']),
                                          pd.DataFrame(data=fa.structure_, columns=factor_list)],
                                          axis=1)
                to_return['df_structure'] = df_structure.to_json(orient='records')
        # return {'Factor Loadings matrix': fa.loadings_.tolist(),
        #         'Original Correlation Matrix': fa.corr_.tolist(),
        #         'Structure Loading Matrix': fa.structure_.tolist(),
        #         'Rotation Matrix': fa.rotation_matrix_.tolist(),
        #        'Communalities': fa.get_communalities().tolist(),
        #        'Original Eigenvalues': original_eigen_values.tolist(),
        #        'Common Factor Eigenvalues': common_factor_eigen_values.tolist(),
        #        'Factor variances': factor_variance.tolist(),
        #        'Proportional Factor Variance': proportional_factor_variance.tolist(),
        #        'Cumulative Factor Variance': cumulative_variance.tolist(),
        #        'uniquenesses from the factor loading matrix': uniquenesses.tolist(),
        #        'Factor scores for a new dataset': new_dataset.tolist()}
        # to_return.update({'df_rotation': df_rotation.to_json(orient='records')})
    # elif rotation == 'oblique':
    #     return {'Factor Loadings matrix': fa.loadings_.tolist(),
    #             'Original Correlation Matrix': fa.corr_.tolist(),
    #             'Structure Loading Matrix': fa.structure_.tolist(),
    #             'Factor Correlations Matrix': fa.phi_.tolist(),
    #             'Rotation Matrix': fa.rotation_matrix_.tolist(),
    #            'Communalities': fa.get_communalities().tolist(),
    #            'Original Eigenvalues': original_eigen_values.tolist(),
    #            'Common Factor Eigenvalues': common_factor_eigen_values.tolist(),
    #            'Factor variances': factor_variance.tolist(),
    #            'Proportional Factor Variance': proportional_factor_variance.tolist(),
    #            'Cumulative Factor Variance': cumulative_variance.tolist(),
    #            'uniquenesses from the factor loading matrix': uniquenesses.tolist(),
    #            'Factor scores for a new dataset': new_dataset.tolist()}

    # else:
        # return {'Factor Loadings matrix': fa.loadings_.tolist(),
        #         'Original Correlation Matrix': fa.corr_.tolist(),
        #         'Rotation Matrix': fa.rotation_matrix_.tolist(),
        #        'Communalities': fa.get_communalities().tolist(),
        #        'Original Eigenvalues': original_eigen_values.tolist(),
        #        'Common Factor Eigenvalues': common_factor_eigen_values.tolist(),
        #        'Factor variances': factor_variance.tolist(),
        #        'Proportional Factor Variance': proportional_factor_variance.tolist(),
        #        'Cumulative Factor Variance': cumulative_variance.tolist(),
        #        'uniquenesses from the factor loading matrix': uniquenesses.tolist(),
        #        'Factor scores for a new dataset': new_dataset.tolist()}

    return to_return


@router.get("/adequacy_test_factor_analysis_bartlett")
async def compute_adequacy_test_bartlett(workflow_id: str,
                                         step_id: str,
                                         run_id: str,
                                         independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    chi_square_value, p_value = calculate_bartlett_sphericity(dataset)

    if p_value < 0.05:
        return {'p-value': p_value, "Interpretation": "Bartlett’s test of sphericity checks whether or not the observed variables intercorrelate at all using the observed correlation matrix against the identity matrix. If the test found statistically insignificant, you should not employ a factor analysis.",
                'Interpretation of this result': "The test was statistically significant, indicating that the observed correlation matrix is not an identity matrix.",
                'chi_square_value': chi_square_value}
    else:
        return {'p-value': p_value,
                "Interpretation": "Bartlett’s test of sphericity checks whether or not the observed variables intercorrelate at all using the observed correlation matrix against the identity matrix. If the test found statistically insignificant, you should not employ a factor analysis.",
                'Interpretation of this result': "The test was statistically insignificant, indicating that the observed correlation matrix is an identity matrix.",
                'chi_square_value': chi_square_value}

@router.get("/adequacy_test_factor_analysis_kmo")
async def compute_adequacy_test_kmo(workflow_id: str,
                                    step_id: str,
                                    run_id: str,
                                    independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    kmo_all,kmo_model = calculate_kmo(dataset)

    if kmo_model < 0.6:
        return {'KMO score per item': kmo_all.tolist(),
                'Interpretation': 'This statistic represents the degree to which each observed variable is predicted, without error, by the other variables in the dataset. In general, a KMO < 0.6 is considered inadequate.',
                'Overall KMO score': kmo_model,
                'Interpretation here': 'Inadequate'}
    else:
        return {'KMO score per item': kmo_all.tolist(),
                'Interpretation': 'This statistic represents the degree to which each observed variable is predicted, without error, by the other variables in the dataset. In general, a KMO < 0.6 is considered inadequate.',
                'Overall KMO score': kmo_model,
                'Interpretation here': 'The overall KMO for our data is greater than 0.6, which is excellent. This value indicates that you can proceed with your planned factor analysis.'}



@router.get("/calculate_confirmatory_factor_analysis")
async def compute_confirmatory_factor_analysis(workflow_id: str,
                                               step_id: str,
                                               run_id: str,
                                               use_smc : bool | None = Query(default=True),
                                               n_factors: int | None = Query(default=3),
                                               rotation : str | None = Query("None",
                                                                             regex="^(None)$|^(varimax)$|^(promax)$|^(oblimin)$|^(oblimax)$|^(quartimin)$|^(quartimax)$|^(equamax)$|^(geomin_obl)$|^(geomin_ort)$"),
                                               method: str | None = Query("minres",
                                                                          regex="^(minres)$|^(ml)$|^(principal)$"),
                                               impute: str | None = Query("drop",
                                                                          regex="^(drop)$|^(mean)$|^(median)$"),
                                               independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)


    if rotation == str(None):
        fa = FactorAnalyzer(n_factors=n_factors, rotation=None, method=method, use_smc=use_smc, impute=impute)
    else:
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method, use_smc=use_smc, impute=impute)


    fa.fit(dataset)

    loadings = fa.loadings_
    specification = ModelSpecification(loadings=loadings, n_factors=n_factors, n_variables=len(dataset.columns))

    cfa = ConfirmatoryFactorAnalyzer(specification, disp=False)
    cfa.fit(dataset)

    print(cfa.get_standard_errors())


@router.get("/mediation_analysis")
async def analysis_mediation(workflow_id: str,
                             step_id: str,
                             run_id: str,
                             dependent_1: str,
                             exposure: str,
                             mediator: str,
                             independent_1: list[str] | None = Query(default=None),
                             independent_2: list[str] | None = Query(default=None)):

    # data = pd.read_csv('example_data/mescobrad_dataset.csv')
    data = load_file_csv_direct(workflow_id, run_id, step_id)
    z = dependent_1 + "~"
    for i in range(len(independent_1)):
        z = z + "+" + independent_1[i]


    if mediator not in z:
        z = z + "+" + mediator

    if exposure not in z:
        z = z + "+" + exposure
    outcome_model = sm.GLM.from_formula(z, data)

    z = mediator + "~"
    for i in range(len(independent_2)):
        z = z + "+" + independent_2[i]

    if exposure not in z:
        z = z + "+" + exposure

    mediator_model = sm.OLS.from_formula(z, data)
    med = Mediation(outcome_model, mediator_model, exposure, mediator).fit()

    print(med.summary())


@router.get("/canonical_correlation_analysis")
async def canonical_correlation(workflow_id: str,
                                step_id: str,
                                run_id: str,
                                n_components: int | None = Query(default=2),
                                independent_variables_1: list[str] | None = Query(default=None),
                                independent_variables_2: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)

    X = dataset[dataset.columns.intersection(independent_variables_1)]
    Y =dataset[dataset.columns.intersection(independent_variables_2)]

    # First, let’s see if there is any correlation between the features of this dataset.
    corr_XY = pd.concat([X,Y],axis=1, join='inner').corr()
    plt.figure(figsize=(5, 5))
    sns.heatmap(corr_XY, cmap='coolwarm', annot=True, linewidths=1, vmin=-1)
    plt.savefig(path_to_storage + "/output/CCA_XYcorr.svg", format="svg")

    # Number of components to keep. Should be in [1, min(n_samples, n_features, n_targets)].
    if n_components > min(X.shape[0], len(independent_variables_1), len(independent_variables_2)):
        n_components = min(X.shape[0], len(independent_variables_1), len(independent_variables_2))

    my_cca = CCA(n_components=n_components)
    # Fit the model
    my_cca.fit(X, Y)
    X_c, Y_c = my_cca.transform(X, Y)

    # Now let’s check if there is any dependency between our canonical variates.
    comp_corr = [np.corrcoef(X_c[:, i], Y_c[:, i])[1][0] for i in range(n_components)]
    comp_titles = ['Comp'+ str(i+1) for i in range(n_components)]
    plt.figure(figsize=(5, 5))
    plt.bar(comp_titles, comp_corr, color='lightgrey', width=0.8, edgecolor='k')
    plt.savefig(path_to_storage + "/output/CCA_comp_corr.svg", format="svg")

    fig, axs = plt.subplots(1, n_components, figsize=(n_components*8, 8), sharey='row')
    for i in range(n_components):
        axs[i].scatter(X_c[:, i], Y_c[:, i], marker="s", label='Comp'+ str(i+1))
        z = np.polyfit(X_c[:, i], Y_c[:, i], 1)
        p = np.poly1d(z)
        axs[i].plot(X_c[:, i], p(X_c[:, i]), color="red", linewidth=3, linestyle="--")
        axs[i].legend(loc='upper left')
        axs[i].set_ylabel('CCY_'+str(i+1), fontsize=14)
        axs[i].set_xlabel('CCX_'+str(i+1), fontsize=14)
        axs[i].set_title('Comp'+str(i+1)+' , corr = %.2f' %
                  np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1])
    plt.savefig(path_to_storage + "/output/CCA_XY_c_corr.svg", format="svg")

    coef_df = pd.DataFrame(np.round(my_cca.coef_, 5), columns=[Y.columns])
    coef_df.index = X.columns
    print(coef_df)
    plt.figure(figsize=(5, 5))
    s= sns.heatmap(coef_df, cmap='coolwarm', annot=True, linewidths=1, vmin=-1)
    s.set(xlabel='Y samle', ylabel='X sample')
    # plt.title = "CCA coefficients."
    plt.savefig(path_to_storage + "/output/CCA_coefs.svg", format="svg")
    plt.show()

    xweights = pd.DataFrame(my_cca.x_weights_, columns=comp_titles)
    xweights.insert(loc=0, column='Feature', value=independent_variables_1)
    print(xweights)
    yweights = pd.DataFrame(my_cca.y_weights_, columns=comp_titles)
    yweights.insert(loc=0, column='Feature', value=independent_variables_2)
    xloadings = pd.DataFrame(my_cca.x_loadings_, columns=comp_titles)
    xloadings.insert(loc=0, column='Feature', value=independent_variables_1)
    yloadings = pd.DataFrame(my_cca.y_loadings_, columns=comp_titles)
    yloadings.insert(loc=0, column='Feature', value=independent_variables_2)
    xrotations = pd.DataFrame(my_cca.x_rotations_, columns=comp_titles)
    xrotations.insert(loc=0, column='Feature', value=independent_variables_1)
    yrotations = pd.DataFrame(my_cca.y_rotations_, columns=comp_titles)
    yrotations.insert(loc=0, column='Feature', value=independent_variables_2)
    Xc_df = pd.DataFrame(X_c, columns=comp_titles)
    Yc_df = pd.DataFrame(Y_c, columns=comp_titles)

    return {'xweights': xweights.to_json(orient='records'),
            'yweights': yweights.to_json(orient='records'),
            'xloadings': xloadings.to_json(orient='records'),
            'yloadings': yloadings.to_json(orient='records'),
            'xrotations': xrotations.to_json(orient='records'),
            'yrotations': yrotations.to_json(orient='records'),
            'coef_df': coef_df.to_json(orient='records'),
            'Xc_df': Xc_df.to_json(orient='records'),
            'Yc_df': Yc_df.to_json(orient='records')}
    # return {'The left singular vectors of the cross-covariance matrices of each iteration.': my_cca.x_weights_.tolist(),
    #         'The right singular vectors of the cross-covariance matrices of each iteration.': my_cca.y_weights_.tolist(),
    #         'The loadings of X.': my_cca.x_loadings_.tolist(),
    #         'The loadings of Y.': my_cca.y_loadings_.tolist(),
    #         'The projection matrix used to transform X.': my_cca.x_rotations_.tolist(),
    #         'The projection matrix used to transform Y.': my_cca.y_rotations_.tolist(),
    #         'The coefficients of the linear model.': my_cca.coef_.tolist(),
    #         'Transformed X': X_c.tolist(),
    #         'Transformed Y': Y_c.tolist()}

@router.get("/granger_analysis")
async def compute_granger_analysis(workflow_id: str,
                                   step_id: str,
                                   run_id: str,
                                   num_lags: int,
                                   predictor_variable: str,
                                   response_variable: str,
                                   all_lags_up_to : bool | None = Query(default=False)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    if all_lags_up_to==False:
        print(grangercausalitytests(dataset[[response_variable, predictor_variable]], maxlag=[num_lags]))
    else:
        print(grangercausalitytests(dataset[[response_variable, predictor_variable]], maxlag=num_lags))

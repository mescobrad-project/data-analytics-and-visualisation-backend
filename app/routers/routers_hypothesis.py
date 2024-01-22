import numpy as np
import pandas as pd
import json
import random
import sklearn.preprocessing
import itertools
from sklearn.cross_decomposition import CCA
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import FastICA
from sklearn.preprocessing import LabelEncoder
import re
from pandas.api.types import is_numeric_dtype
from sphinx.addnodes import index
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import grangercausalitytests, acf
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.utils import corr, cov
from factor_analyzer.confirmatory_factor_analyzer import ModelSpecification, ConfirmatoryFactorAnalyzer
from factor_analyzer import FactorAnalyzer
from scipy.stats import jarque_bera, fisher_exact, ranksums, chisquare, kruskal, alexandergovern, kendalltau, f_oneway, \
    shapiro, \
    kstest, anderson, normaltest, boxcox, yeojohnson, bartlett, levene, fligner, obrientransform, pearsonr, spearmanr, \
    pointbiserialr, ttest_ind, mannwhitneyu, wilcoxon, ttest_rel, skew, kurtosis, probplot, zscore, t
from typing import Optional, Union, List
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.mediation import Mediation
import statsmodels.stats.api as sms
from pydantic import BaseModel
from statsmodels.stats.diagnostic import het_goldfeldquandt
from fastapi import FastAPI, Path, Query, APIRouter
from fastapi.responses import JSONResponse
import pingouin
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from lifelines.utils import to_episodic_format
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, SGDClassifier, HuberRegressor,Lars, PoissonRegressor, LogisticRegression
from sklearn.svm import SVR, LinearSVR, LinearSVC
from pingouin import ancova, mediation_analysis
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

from app.routers.routers_communication import task_complete
# from app.pydantic_models import ModelMultipleComparisons
from app.utils.utils_datalake import fget_object, get_saved_dataset_for_Hypothesis, upload_object
from app.utils.utils_general import get_local_storage_path, get_single_file_from_local_temp_storage, load_data_from_csv, \
    load_file_csv_direct, get_all_files_from_local_temp_storage, write_function_data_to_config_file
from tabulate import tabulate
import seaborn as sns
from datetime import datetime
import os
from os.path import isfile, join
import math
from sklearn.preprocessing import StandardScaler

from app.utils.utils_hypothesis import create_plots, compute_skewness, outliers_removal, compute_kurtosis, \
    statisticsMean, statisticsMin, statisticsMax, statisticsStd, statisticsCov, statisticsVar, statisticsStandardError, \
    statisticsConfidenceLevel, DataframeImputation
from semopy import Model, estimate_means, ModelMeans, semplot, calc_stats, gather_statistics, Optimizer, efa
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
router = APIRouter()
# data = pd.read_csv('example_data/mescobrad_dataset.csv')
# data = data.drop(["Unnamed: 0"], axis=1)
# data = pd.read_csv('example_data/sample_questionnaire.csv')

def normality_test_content_results(column: str, selected_dataframe,path_to_storage:str):
    try:
        if (selected_dataframe[column].dtypes == 'float64' or selected_dataframe[column].dtypes =='int64'):
            # Creating Box-plot
            html_str_B = create_plots(plot_type='BoxPlot', column=column,second_column='', selected_dataframe=selected_dataframe, path_to_storage=path_to_storage, filename='BoxPlot')
            # Creating QQ-plot
            html_str = create_plots(plot_type='QQPlot', column=column, second_column='', selected_dataframe=selected_dataframe, path_to_storage=path_to_storage, filename='QQPlot')
            # Creating Probability-plot
            html_str_P = ''
            # html_str_P = create_plots(plot_type='PPlot', column=column, second_column='', selected_dataframe=selected_dataframe, path_to_storage=path_to_storage, filename='PPlot')
            #Creating histogram
            html_str_H = create_plots(plot_type='HistogramPlot', column=column, second_column='', selected_dataframe=selected_dataframe, path_to_storage=path_to_storage, filename='HistogramPlot')
            skewtosend = compute_skewness(column, selected_dataframe)
            kurtosistosend = compute_kurtosis(column, selected_dataframe)
            st_dev = statisticsStd(column, selected_dataframe,1)
            sample_variance = statisticsVar(column, selected_dataframe)
            standard_error = statisticsStandardError(column, selected_dataframe)
            # Used Statistics lib for cross-checking
            # standard_deviation = statistics.stdev(data[str(column)])
            median_value = float(np.percentile(selected_dataframe[str(column)], 50)) if not math.isnan(float(np.percentile(selected_dataframe[str(column)], 50))) else ''
            # Used a different way to calculate Median
            # TODO: we must investigate why it returns a different value
            # med2 = np.median(data[str(column)])
            mean_value = np.mean(selected_dataframe[str(column)]) if not np.isinf(np.mean(selected_dataframe[str(column)])) else ''
            num_rows = selected_dataframe[str(column)].shape
            top5 = sorted(selected_dataframe[str(column)].tolist(), reverse=True)[:5]
            last5 = sorted(selected_dataframe[str(column)].tolist(), reverse=True)[-5:]

            confidence_level = statisticsConfidenceLevel(column, selected_dataframe,0.95)
            return {'plot_column': column, 'qqplot': html_str, 'histogramplot': html_str_H, 'boxplot': html_str_B, 'probplot': html_str_P, 'skew': skewtosend, 'kurtosis': kurtosistosend, 'standard_deviation': st_dev,'standard_error':standard_error, "median": median_value, "mean": mean_value, "sample_N": num_rows, "top_5": top5, "last_5": last5, 'sample_variance':sample_variance,'confidence_level':confidence_level}
        else:
            print('The type of:' + column +' is:'+ str(selected_dataframe[column].dtypes))
            raise Exception
    except Exception as e:
        print('normality_test_content_results  ' +e.__str__())
        return -1

def transformation_extra_content_results(column_In: str, column_Out:str, selected_dataframe,path_to_storage:str):
    try:
        if (selected_dataframe[column_In].dtypes == 'float64' or selected_dataframe[column_In].dtypes == 'int64'):
            # fig = plt.figure()
            # plt.plot(selected_dataframe[str(column_In)], selected_dataframe[str(column_In)],
            #          color='blue', marker="*")
            # plt.plot(selected_dataframe[str(column_Out)], selected_dataframe[str(column_In)],
            #          color='red', marker="o")
            fig = px.scatter(selected_dataframe, x=str(column_In), y=str(column_Out), labels={'x':"Original Values", 'y':"Transformed Values"})
            fig.write_image(path_to_storage + "/output/ComparisonPlot.svg")
            html_str = fig.to_json(pretty=True)

            # plt.title("Transformed data Comparison")
            # plt.xlabel("Original Values")
            # plt.ylabel("Transformed Values")
            # plt.savefig(path_to_storage + "/output/ComparisonPlot.svg", format="svg")
            # plt.show()
            # html_str_Transf = mpld3.fig_to_html(fig)
            return html_str
        else:
            raise Exception
    except Exception as e:
        print('transformation_extra_content_results '+ e)
        return -1

class FunctionOutputItem(BaseModel):
    """
    Known metadata information
    "files" : [["run_id: "string" , "step_id": "string"], "output":"string"]
     """
    workflow_id:str
    run_id: str
    step_id: str
    # file: str

@router.get("/return_all_files")
async def return_all_files(workflow_id: str, step_id: str, run_id: str):
    try:
        list_of_files = get_all_files_from_local_temp_storage(workflow_id, run_id, step_id)
    except Exception as e:
        print(e)
        print("Error : Failed to retrieve file names")
        return []
    return {'files': list_of_files}


@router.put("/save_hypothesis_output")
async def save_hypothesis_output(item: FunctionOutputItem) -> dict:
    try:
        path_to_storage = get_local_storage_path(item.workflow_id, item.run_id, item.step_id)
        files_to_upload = [f for f in os.listdir(path_to_storage + '/output') if isfile(join(path_to_storage + '/output', f))]
        for file in files_to_upload:
            out_filename = path_to_storage + '/output/' + file
            upload_object(bucket_name="demo", object_name='expertsystem/workflow/'+ item.workflow_id+'/'+ item.run_id+'/'+
                                                          item.step_id+'/analysis_output/' + file, file=out_filename)
        return JSONResponse(content='info.json file has been successfully uploaded to the DataLake', status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content='Error in saving info.json object to the DataLake',status_code=501)

@router.get("/return_dataset")
async def dataset_content(workflow_id: str, step_id: str, run_id: str, file_name:str):
    try:
        path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
        name_of_files = get_all_files_from_local_temp_storage(workflow_id, run_id, step_id)
        if file_name in name_of_files:
            data = load_data_from_csv(path_to_storage + "/" + file_name)
            df = pd.DataFrame(data.describe())
            df.insert(0,'Index',df.index)
            # df['Index']=df.index
            # Add data.info()
            df1 = pd.DataFrame()
            df1['Non Null Count'] = data.notna().sum()
            df1['Dtype'] = data.dtypes
            dfinfo =df1.T
            dfinfo['Index']=dfinfo.index
            df = pd.concat([df, dfinfo], ignore_index=True)
        else:
            print("Error : Failed to find the file")
            return {'dataFrame': {}}
        return {'dataFrame': df.to_json(orient='records', default_handler=str)}
    except Exception as e:
        print(e)
        return JSONResponse(content='Error : Failed to retrieve column names', status_code=501)


@router.get("/return_columns")
async def name_columns(workflow_id: str, step_id: str, run_id: str, file_name:str|None=None):
    try:
        if file_name is None:
            # path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
            # name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
            # data = load_data_from_csv(path_to_storage + "/" + name_of_file)
            return {'columns': []}
        else:
            path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
            name_of_files = get_all_files_from_local_temp_storage(workflow_id, run_id, step_id)
            print(name_of_files)
            print(file_name)
            if file_name in name_of_files:
                data = load_data_from_csv(path_to_storage + "/" + file_name)
            else:
                print("Error : Failed to find the file")
                return {'columns': []}

        columns = data.columns
        return{'columns': list(columns)}
    except Exception as e:
        print(e)
        return JSONResponse(content='Error : Failed to retrieve column names',status_code=501)

@router.get("/return_cox_columns")
async def return_cox_columns(workflow_id: str, step_id: str, run_id: str, file_name:str|None=None):
    try:
        if file_name is None:
            # path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
            # name_of_file = get_single_file_from_local_temp_storage(workflow_id, run_id, step_id)
            # data = load_data_from_csv(path_to_storage + "/" + name_of_file)
            return {'columns': []}
        else:
            path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
            name_of_files = get_all_files_from_local_temp_storage(workflow_id, run_id, step_id)
            print(name_of_files)
            print(file_name)
            if file_name in name_of_files:
                data = load_data_from_csv(path_to_storage + "/" + file_name)
            else:
                print("Error : Failed to find the file")
                return {'columns': []}

        columns = data.columns

        suitable_columns = []
        for column in columns:
            if len(pd.unique(data[column])) <= 10:
                suitable_columns.append(column)
        return{'columns': suitable_columns}
    except Exception as e:
        print(e)
        return JSONResponse(content='Error : Failed to retrieve column names',status_code=501)


@router.get("/return_binary_columns")
async def name_columns(workflow_id: str, step_id: str, run_id: str, file_name:str|None=None):
    try:
        print(file_name)
        if file_name is None:
            print("Error : Failed to find the file")
            return {'columns': []}
        else:
            path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
            name_of_files = get_all_files_from_local_temp_storage(workflow_id, run_id, step_id)
            if file_name in name_of_files:
                data = load_data_from_csv(path_to_storage + "/" + file_name)
            else:
                print("Error : Failed to find the file")
                return {'columns': []}
        print(data.columns)
        for b_column in data.columns:
            if data[b_column].unique().shape[0] > 2:
                data = data.drop([b_column], axis=1)

        columns = data.columns
        return{'columns': list(columns)}
    except Exception as e:
        print(e)
        return JSONResponse(content='Error : Failed to retrieve binary-column names',status_code=501)

# TODO: Delete this router, if we don't need it anymore
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

    dfv = pd.DataFrame()
    # path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = [column]
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        test_status='Unable to retrieve datasets'

        data = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        column = dfv['Variable'][0]
        # Remove Nans for the calculations
        data = data.dropna(subset=[str(column)])

        results_to_send = normality_test_content_results(column, data, path_to_storage)
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
        # Prepare content for info.json
        test_status = 'Unable to compute ' + name_test + \
                      ' for the selected columns. NaNs or nonnumeric values are selected.'

        if results_to_send == -1:
            results_to_send = {'plot_column': "", 'qqplot': "", 'histogramplot': "", 'boxplot': "", 'probplot': "",
             'skew': 0, 'kurtosis': 0, 'standard_deviation': 0, 'standard_error':0, "median": 0,
             "mean": 0, "sample_N": 0, "top_5": [], "last_5": [],'sample_variance':0, 'confidence_level':0}
            raise Exception
        new_data = {
            "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            "workflow_id": workflow_id,
            "run_id": run_id,
            "step_id": step_id,
            "test_name": 'Normality test',
            "test_params": {
                'selected_method': name_test,
                'selected_variable': column,
                'alternative': alternative,
                'nan_policy': nan_policy},
            "test_results": {
                'skew': results_to_send['skew'],
                'kurtosis': results_to_send['kurtosis'],
                'standard_deviation': results_to_send['standard_deviation'],
                'sample_variance':results_to_send['sample_variance'],
                'standard_error':results_to_send['standard_error'],
                'median': results_to_send['median'],
                'mean': results_to_send['mean'],
                'confidence_level': results_to_send['confidence_level'],
                'sample_N': results_to_send['sample_N'],
                'top_5': results_to_send['top_5'],
                'last_5': results_to_send['last_5']
            },
            'Output_datasets': [],
            'Saved_plots': [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                             step_id + '/BoxPlot.svg'},
                                    {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                             step_id + '/PPlot.svg'},
                                    {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                             step_id + '/HistogramPlot.svg'},
                                    {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                             step_id + '/QQPlot.svg'}
                                    ]
        }

        if name_test == 'Shapiro-Wilk':
            shapiro_test = shapiro(data[str(column)])
            descr = 'Sample looks Gaussian (fail to reject H0)' if shapiro_test.pvalue > 0.05 else 'Sample does not look Gaussian (reject H0)'
            statistic = shapiro_test.statistic
            p_value = shapiro_test.pvalue
        elif name_test == 'Kolmogorov-Smirnov':
            ks_test = kstest(data[str(column)], 'norm', alternative=alternative)
            descr = 'Sample looks Gaussian (fail to reject H0)' if ks_test.pvalue > 0.05 else 'Sample does not look Gaussian (reject H0)'
            statistic = ks_test.statistic
            p_value = ks_test.pvalue
        elif name_test == 'Anderson-Darling':
            anderson_test = anderson(data[str(column)])
            list_anderson = []
            for i in range(len(anderson_test.critical_values)):
                sl, cv = anderson_test.significance_level[i], anderson_test.critical_values[i]
                if anderson_test.statistic < anderson_test.critical_values[i]:
                    list_anderson.append('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
                else:
                    list_anderson.append('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
            with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
                file_data = json.load(f)
                new_data['test_results']|= {
                    'statistic': anderson_test.statistic, 'critical_values': list(anderson_test.critical_values), 'significance_level': list(anderson_test.significance_level), 'Description': list_anderson}
                file_data['results'] = new_data
                f.seek(0)
                json.dump(file_data, f, indent=4)
                f.truncate()
            return JSONResponse(content={'status': 'Success','statistic':anderson_test.statistic, 'critical_values': list(anderson_test.critical_values), 'significance_level': list(anderson_test.significance_level), 'Description': list_anderson, 'results': results_to_send}, status_code=200)
        elif name_test == 'D’Agostino’s K^2':
            stat, p = normaltest(data[str(column)], nan_policy=nan_policy)
            descr = 'Sample looks Gaussian (fail to reject H0)' if p > 0.05 else 'Sample does not look Gaussian (reject H0)'
            statistic = stat
            p_value = p
        elif name_test == 'Jarque-Bera':
            jarque_bera_test = jarque_bera(data[str(column)])
            statistic = jarque_bera_test.statistic
            p_value = jarque_bera_test.pvalue
            descr = 'Sample looks Gaussian (fail to reject H0)' if p_value > 0.05 else 'Sample does not look Gaussian (reject H0)'

        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            new_data['test_results']|= {
                'statistic': statistic, 'p_value': p_value, 'Description': descr}
            file_data['results'] = new_data
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success','statistic': statistic, 'p_value': p_value, 'Description': descr, 'results': results_to_send}, status_code=200)
    except Exception as e:
        # df["Error"] = ["Unable to conduct Normality test"]
        print(e)
        return JSONResponse(content={'status':test_status,'statistic': "", 'p_value': "", 'Description': "", 'results': {}, 'critical_values': [], 'significance_level':[]}, status_code=200)

@router.get("/transform_data", tags=['hypothesis_testing'])
async def transform_data(workflow_id: str,
                         step_id: str,
                         run_id: str,
                         column: str,
                         name_transform: str | None = Query("Box-Cox",
                                                           regex="^(Box-Cox)$|^(Yeo-Johnson)$|^(Log)$|^(Squared-root)$|^(Cube-root)$"),
                         lmbd: Optional[float] = None,
                         alpha: Optional[float] = None) -> dict:
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = [column]
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        test_status='Unable to retrieve datasets'
        # We expect only one here
        data = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        column = dfv['Variable'][0]
        test_status = 'Unable to compute ' + name_transform + \
                      ' for the selected columns.'

        newColumnName = "Transf_" + column
        if name_transform == 'Box-Cox':
            if lmbd == None:
                if alpha == None:
                    boxcox_array, maxlog = boxcox(np.array(data[str(column)]))
                    data[newColumnName] = boxcox_array
                else:
                    boxcox_array, maxlog, z = boxcox(np.array(data[str(column)]), alpha=alpha)
                    data[newColumnName] = boxcox_array
            else:
                if alpha == None:
                    y = boxcox(np.array(data[str(column)]), lmbda=lmbd)
                    data[newColumnName] = y
                else:
                    y = boxcox(np.array(data[str(column)]), lmbda=lmbd, alpha=alpha)
                    data[newColumnName] = y
        elif name_transform == 'Yeo-Johnson':
            if lmbd == None:
                yeojohnson_array, maxlog = yeojohnson(np.array(data[str(column)]))
                data[newColumnName] = yeojohnson_array
            else:
                yeojohnson_array = yeojohnson(np.array(data[str(column)]), lmbda=lmbd)
                data[newColumnName] = yeojohnson_array
        elif name_transform == 'Log':
            log_array = np.log(data[str(column)])
            data[newColumnName] = log_array
        elif name_transform == 'Squared-root':
            sqrt_array = np.sqrt(data[str(column)])
            data[newColumnName] = sqrt_array
        elif name_transform == 'Cube-root':
            cbrt_array = np.cbrt(data[str(column)])
            data[newColumnName] = cbrt_array

        # Remove inf
        data[newColumnName] = data[newColumnName].replace([np.inf, -np.inf], np.nan)
        data.to_csv(path_to_storage + '/output/new_dataset.csv', index=False)
        # Remove Nans for the calculations
        data = data.dropna(subset=[str(newColumnName)])

        results_to_send = normality_test_content_results(newColumnName, data, path_to_storage)
        if results_to_send == -1:
            results_to_send = {'plot_column': "", 'qqplot': "", 'histogramplot': "", 'boxplot': "", 'probplot': "",
                               'skew': 0, 'kurtosis': 0, 'standard_deviation': 0, "median": 0,
                               "mean": 0, "sample_N": 0, "top_5": [], "last_5": []}
            raise Exception
        results_to_send_extra = transformation_extra_content_results(column, newColumnName, data, path_to_storage)
        if results_to_send_extra== -1:
            results_to_send['transf_plot']=''
            raise Exception
        results_to_send['transf_plot'] = results_to_send_extra
        # Prepare content for info.json
        new_data = {
            "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            "workflow_id": workflow_id,
            "run_id": run_id,
            "step_id": step_id,
            "test_name": 'Transormation test',
            "test_params": {
                'selected_method': name_transform,
                'selected_variable': column,
                'lmbd': lmbd,
                'alpha': alpha},
            "test_results": {
                'skew': results_to_send['skew'],
                'kurtosis': results_to_send['kurtosis'],
                'standard_deviation': results_to_send['standard_deviation'],
                'median': results_to_send['median'],
                'mean': results_to_send['mean'],
                'standard_error': results_to_send['standard_error'],
                'sample_variance': results_to_send['sample_variance'],
                'confidence_level': results_to_send['confidence_level'],
                'sample_N': results_to_send['sample_N'],
                'top_5': results_to_send['top_5'],
                'last_5': results_to_send['last_5']
            },
            'Output_datasets': [{"file": 'workflows/'+ workflow_id+'/'+ run_id+'/'+
                                         step_id+'/new_dataset.csv'}],
            'Saved_plots': [{"file": 'workflows/'+ workflow_id+'/'+ run_id+'/'+
                                         step_id+'/BoxPlot.svg'},
                            {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                     step_id + '/PPlot.svg'},
                            {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                     step_id + '/HistogramPlot.svg'},
                            {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                     step_id + '/QQPlot.svg'},
                            {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                     step_id + '/Scatter_Two_Variables.svg'},
                            ]
        }
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= new_data
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()

        return JSONResponse(content={'status': 'Success','transformed array': data[newColumnName].to_json(orient='records'), 'data': {}, 'results': results_to_send}, status_code=200)
        # return JSONResponse(content={'status': 'Success','transformed array': data[newColumnName].to_json(orient='records'), 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status':test_status +'\n'+ e.__str__(),'transformed array': {}, 'data': {}, 'results': {}}, status_code=200)


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
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = [column_1, column_2]
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        test_status='Unable to retrieve datasets'
        # We expect only one here
        data = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        column_1 = dfv['Variable'][0]
        column_2 = dfv['Variable'][1]
        test_status = 'Unable to compute Point Biserial correlation for the selected columns. NaNs or nonnumeric values are selected.'

        le = LabelEncoder()
        new_column_1 = 'le_'+str(column_1)
        data[new_column_1] = le.fit_transform(data[str(column_1)])
        if not pd.to_numeric(data[str(column_2)], errors='coerce').notnull().all():
            raise Exception

        unique_values = np.unique(data[new_column_1])
        unique_values.sort()
        if len(unique_values) == 2:
            html_scr = create_plots(plot_type='Scatter_Two_Variables', column=new_column_1, second_column=column_2, selected_dataframe=data, path_to_storage=path_to_storage, filename='Scatter_Two_Variables')
            sub_set_a = data[data[new_column_1] != unique_values[1]]
            sub_set_b = data[data[new_column_1] != unique_values[0]]
            new_dataset_for_bp = [sub_set_a[str(column_2)], sub_set_b[str(column_2)]]
            html_box = create_plots(plot_type='BoxPlot', column=column_2, second_column=new_column_1, selected_dataframe=new_dataset_for_bp, path_to_storage=path_to_storage, filename='BoxPlot')
            html_hist_A = create_plots(plot_type='HistogramPlot', column=column_2, second_column='',
                                       selected_dataframe=sub_set_a, path_to_storage=path_to_storage, filename='HistogramPlot_GroupA')
            html_hist_B = create_plots(plot_type='HistogramPlot', column=column_2, second_column='',
                                       selected_dataframe=sub_set_b, path_to_storage=path_to_storage, filename='HistogramPlot_GroupB')
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
            pointbiserialr_test = pointbiserialr(df[new_column_1], df[str(column_2)])
            df.to_csv(path_to_storage + '/output/new_dataset.csv', index=False)

            data_to_return = {
                    'sample_A': {
                        'value': str(unique_values[0]),
                        'N': len(sub_set_a),
                        'N_clean':  len(sub_set_a_clean),
                        'outliers': outliers_a[column_2].to_json(orient='records'),
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
                        'Norm_statistic': shapiro_test_B.statistic,
                        'Norm_p_value': shapiro_test_B.pvalue
                    },
                    'correlation': pointbiserialr_test[0],
                    'p_value': pointbiserialr_test[1]
                    }

            with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
                # Load existing data into a dict.
                file_data = json.load(f)
                # Join new data
                new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": 'Point Biserial Correlation',
                    "test_params": {'Binary variable': str(column_1),
                                    'Variable': str(column_2)},
                    "test_results": data_to_return,
                    "Output_datasets":[{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                step_id + '/new_dataset.csv'}],
                    "Saved_plots": [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                             step_id + '/BoxPlot.svg'},
                                    {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                             step_id + '/HistogramPlot_GroupA.svg'},
                                    {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                             step_id + '/HistogramPlot_GroupB.svg'},
                                    {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                             step_id + '/Scatter_Two_Variables.svg'}]
                    }
                file_data['results'] |= new_data
                f.seek(0)
                json.dump(file_data, f, indent=4)
                f.truncate()
            data_to_return['new_dataset'] = df.to_json(orient='records')
            data_to_return['status'] = 'Success'
            return data_to_return
            return JSONResponse(content={data_to_return}, status_code=200)
        else:
            test_status = 'Dichotomus Variable must be selected.'
            raise Exception
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'sample_A': {
                        'value': '',
                        'N': '',
                        'N_clean':  '',
                        'outliers': '',
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
                        'Norm_statistic': '',
                        'Norm_p_value': ''
                    },
                    'correlation': '',
                    'p_value': '',
                    'new_dataset': []}, status_code=200)


@router.get("/check_homoscedasticity", tags=['hypothesis_testing'])
async def check_homoskedasticity(workflow_id: str,
                                 step_id: str,
                                 run_id: str,
                                 columns: list[str] | None = Query(default=None),
                                 name_of_test: str | None = Query("Levene",
                                                                  regex="^(Levene)$|^(Bartlett)$|^(Fligner-Killeen)$"),
                                 center: Optional[str] | None = Query("median",
                                                                      regex="^(trimmed)$|^(median)$|^(mean)$")):
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = columns
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        data = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        columns = dfv['Variable']

        test_status = 'Unable to compute Homoscedasticity for the selected columns. NaNs or nonnumeric values are selected.'

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
        # print(*args)
        if name_of_test == "Bartlett":
            statistic, p_value = bartlett(*args)
        elif name_of_test == "Fligner-Killeen":
            statistic, p_value = fligner(*args, center=center)
        else:
            statistic, p_value = levene(*args, center=center)

        new_data = {
            "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            "workflow_id": workflow_id,
            "run_id": run_id,
            "step_id": step_id,
            "test_name": 'Homoscedasticity test',
            "test_params": {
                'selected_method': name_of_test,
                'selected_variable': columns.to_dict(),
                'center': center
            },
            "test_results": {
                'statistic': statistic,
                'p_value': p_value,
                'variance': var
            },
            'Output_datasets': [],
            'Saved_plots': []
            }
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= new_data
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success','statistic': statistic, 'p_value': p_value, 'variance': var}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status':test_status,'statistic': "", 'p_value': "", 'variance': ""}, status_code=200)


@router.get("/transformed_data_for_use_in_an_ANOVA", tags=['hypothesis_testing'])
async def transform_data_anova(
        workflow_id: str,
        step_id: str,
        run_id: str,
        variables: list[str] | None = Query(default=None)):
    dfv = pd.DataFrame()
    df = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        data = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        variables = dfv['Variable']

        test_status = 'Unable to compute Obrien transformation for the selected columns. NaNs or nonnumeric values are selected.'
        # Keep requested Columns
        selected_columns = pd.unique(dfv['Variable'])
        args = []
        args_name=[]
        for column in data.columns:
            if column not in selected_columns:
                data = data.drop(str(column), axis=1)
            else:
                args.append(data[column])
                args_name.append(column)

        tall = obrientransform(*args)
        df = pd.DataFrame(tall, index=args_name)
        df = df.T
        df.to_csv(path_to_storage + '/output/new_dataset.csv', index=False)

        new_data = {
            "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            "workflow_id": workflow_id,
            "run_id": run_id,
            "step_id": step_id,
            "test_name": 'Obrien Transform test',
            "test_params": {
                'selected_variable': variables.to_dict()
            },
            "test_results": {
            },
            "Output_datasets":[{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                    step_id + '/new_dataset.csv'}],
            'Saved_plots': []
        }

        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= new_data
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success',
                                     'Dataframe': df.to_json(orient="records")},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status':test_status,
                                     'Dataframe': df.to_json(orient="records")},
                            status_code=200)

@router.get("/statistical_tests", tags=['hypothesis_testing'])
async def statistical_tests(workflow_id: str,
                            step_id: str,
                            run_id: str,
                            columns: list[str] | None = Query(default=None),
                            # column_1: str,
                            # column_2: str,
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
    dfv = pd.DataFrame()
    df = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status=''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = columns
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        # We expect only one here
        test_status='Unable to retrieve datasets'
        data = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        columns = dfv['Variable'].tolist()
        selected_columns = pd.unique(dfv['Variable'])
        for column in data.columns:
            if column not in selected_columns:
                data = data.drop(str(column), axis=1)

        test_status = 'Unable to compute ' + statistical_test + \
                      ' for the selected columns. NaNs or nonnumeric values are selected.'
        if statistical_test == "Welch t-test":
            if len(data.columns) != 2:
                test_status = 'Two variables must be selected for '+statistical_test
                raise Exception
            statistic, p_value = ttest_ind(data.iloc[:, 0],data.iloc[:, 1], nan_policy=nan_policy, equal_var=False, alternative=alternative)
        elif statistical_test == "Independent t-test":
            if len(data.columns) != 2:
                test_status = 'Two variables must be selected for ' + statistical_test
                raise Exception
            statistic, p_value = ttest_ind(data.iloc[:, 0],data.iloc[:, 1], nan_policy=nan_policy, alternative=alternative)
        elif statistical_test == "t-test on TWO RELATED samples of scores":
            if len(data.columns) != 2:
                test_status = 'Two variables must be selected for ' + statistical_test
                raise Exception
            elif np.shape(data.iloc[:, 0])[0] != np.shape(data.iloc[:, 1])[0]:
                test_status = 'The arrays must have the same shape for' + statistical_test
                raise Exception
            statistic, p_value = ttest_rel(data.iloc[:, 0],data.iloc[:, 1], nan_policy=nan_policy, alternative=alternative)
        elif statistical_test == "Mann-Whitney U rank test":
            if len(data.columns) != 2:
                test_status = 'Two variables must be selected for ' + statistical_test
                raise Exception
            statistic, p_value = mannwhitneyu(data.iloc[:, 0],data.iloc[:, 1], nan_policy=nan_policy, alternative=alternative, method=method)
        elif statistical_test == "Wilcoxon signed-rank test":
            if len(data.columns) != 2:
                test_status = 'Two variables must be selected for ' + statistical_test
                raise Exception
            elif np.shape(data.iloc[:, 0])[0] != np.shape(data.iloc[:, 1])[0]:
                test_status = 'The arrays must have the same shape for' + statistical_test
                raise Exception
            statistic, p_value = wilcoxon(data.iloc[:, 0],data.iloc[:, 1], alternative=alternative, nan_policy=nan_policy, correction=correction, zero_method=zero_method, mode=mode)
        elif statistical_test == "Alexander Govern test":
            samples = []
            for k in data.columns:
                samples.append(data[k])
            AlexanderGovernResult = alexandergovern(*samples, nan_policy=nan_policy)
            statistic, p_value = AlexanderGovernResult.statistic, AlexanderGovernResult.pvalue
        elif statistical_test == "Kruskal-Wallis H-test":
            samples = []
            for k in data.columns:
                samples.append(data[k])
            statistic, p_value = kruskal(*samples, nan_policy=nan_policy)
        elif statistical_test == "one-way ANOVA":
            samples = []
            for k in data.columns:
                print(k)
                print(data[k])
                samples.append(data[k])
            print(samples)
            statistic, p_value = f_oneway(*samples)
        elif statistical_test == "Wilcoxon rank-sum statistic":
            if len(data.columns) != 2:
                test_status = 'Two variables must be selected for ' + statistical_test
                raise Exception
            statistic, p_value = ranksums(data.iloc[:, 0],data.iloc[:, 1], nan_policy=nan_policy, alternative=alternative)
        elif statistical_test == "one-way chi-square test":
            samples = []
            for k in data.columns:
                samples.append(data[k])
            # TODO: We can have several f_obs columns of observed frequencies and
            #  f_exp column of the expected frequencies
            statistic, p_value = chisquare(*samples)
        # Provide Mean and Std for all cases
        df = pd.DataFrame(data=
                          {"Variable": data.columns,
                           'mean': data.mean(),
                           "standard deviation": data.std()},
                          index=data.columns)
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": statistical_test,
                    "test_params": {
                        'selected_method': method,
                        'selected_variable': columns,
                        'nan_policy': nan_policy,
                        'alternative': alternative,
                        'correction': correction,
                        'mode': mode,
                        'zero_method': zero_method
                    },
                    "test_results": {
                        'statistic': statistic,
                        'p-value': p_value,
                        'Mean & std': df.to_dict()}
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'statistic': statistic,
                                     'p-value': p_value, 'mean_std': df.to_json(orient='records')}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status':test_status, 'statistic': '',
                                     'p-value': '', 'mean_std': df.to_json(orient='records')}, status_code=200)


@router.get("/multiple_comparisons", tags=['hypothesis_testing'])
async def p_value_correction(workflow_id: str,
                             step_id: str,
                             run_id: str,
                             method: str,
                             alpha: float,
                             p_value: list[str] | None = Query(default=None)):
    dfv = pd.DataFrame()
    df = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = p_value
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        # We expect only one here
        test_status = 'Unable to retrieve datasets'
        data = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        # We expect only 1 column
        if len(pd.unique(dfv['Variable'])) != 1:
            test_status = 'Only 1 set of p-values is expected'
            raise Exception

        p_value = dfv['Variable'][0]
        test_status = 'Unable to compute ' + method + ' Multitest for the selected p-values.'
        if method == 'Bonferroni':
            z = multipletests(pvals=data[p_value], alpha=alpha, method='bonferroni')
        elif method == 'sidak':
            z = multipletests(pvals=data[p_value], alpha=alpha, method='sidak')
        elif method == 'benjamini-hochberg':
            z = multipletests(pvals=data[p_value], alpha=alpha, method='fdr_bh')
        elif method == 'benjamini-yekutieli':
            z = multipletests(pvals=data[p_value], alpha=alpha, method='fdr_by')
        else:
            z = multipletests(pvals=data[p_value], alpha=alpha, method= method)

        df['values'] = data[p_value]
        df['rejected'] = [str(x) for x in z[0]]
        df['corrected_p_values'] = z[1]
        df.to_csv(path_to_storage + '/output/new_dataset.csv', index=False)

        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": "Multitesting and adjustment of pvalues",
                    "test_params": {
                        'selected_method': method,
                        'selected_variable': p_value,
                        'alpha': alpha
                    },
                    "test_results": ''
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/'+ workflow_id+'/'+ run_id+'/'+
                                         step_id+'/new_dataset.csv'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return {'status':'Success', 'result': df.to_json(orient='records')}
    except Exception as e:
        print(e)
        return {'status':test_status,'result': df.to_json(orient='records')}


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
    dfv = pd.DataFrame()
    df = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    to_return={'number_of_features': '',
            'features_columns': [],
            'number_of_classes':'',
            'classes_': [],
            'number_of_components': '',
            'explained_variance_ratio': df.to_json(orient='records'),
            'means_': df.to_json(orient='records'),
            'priors_': df.to_json(orient='records'),
            'scalings_': df.to_json(orient='records'),
            'xbar_': df.to_json(orient='records'),
            'coefficients': df.to_json(orient='records'),
            'intercept': df.to_json(orient='records')}
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        independent_variables = dfv['Variable']
        dependent_variable = dependent_variable.split("--")[1]
        selected_columns = pd.unique(dfv['Variable'])

        # We expect only one here
        test_status = 'Unable to retrieve datasets'
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])

        # dataset = load_file_csv_direct(workflow_id, run_id, step_id)
        # dataset = pd.read_csv('example_data/mescobrad_dataset.csv')
        df_label = dataset[str(dependent_variable)]
        for columns in dataset.columns:
            if columns not in selected_columns:
                dataset = dataset.drop(str(columns), axis=1)
        test_status = 'Unable to compute LDA. Variables with numeric values must be selected.'
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
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Linear discriminant analysis',
                "test_params": {'Dependent': dependent_variable,
                                'Independent Variables': list(selected_columns),
                                'solver':solver,
                                'shrinkage': shrinkage_1},
                "test_results": to_return,
                "Output_datasets": [],
                "Saved_plots": []
            }
            file_data['results'] |= new_data
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        print(test_status)
        print(to_return)
        return JSONResponse(content={'status': 'Success', 'result': to_return}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'result': to_return}, status_code=200)

    # return {'coefficients': df_coefs.to_json(orient='split'), 'intercept': df_intercept.to_json(orient='split')}


@router.get("/principal_component_analysis")
async def principal_component_analysis(workflow_id: str,
                                       step_id: str,
                                       run_id: str,
                                       categorical_variable: str,
                                       n_components_1: int | None = Query(default=None),
                                       svd_solver: str | None = Query("auto", regex="^(auto)$|^(full)$"),
                                       independent_variables: list[str] | None = Query(default=None)):
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        independent_variables = dfv["Variable"].tolist()
        categorical_variable = categorical_variable.split("--")[1]
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        categorical_variable_df = dataset[categorical_variable]

        for columns in dataset.columns:
            if columns not in np.append(independent_variables, categorical_variable):
                dataset = dataset.drop(str(columns), axis=1)

        X = dataset.loc[:, independent_variables].values
        if n_components_1 > min(int(np.shape(X)[0]), int(np.shape(X)[1])):
            test_status = 'components must be <= min(n_samples, n_features)'
            raise Exception
        X_norm = StandardScaler().fit_transform(X)  # normalizing the features
        # print(X_norm.shape)
        print(np.mean(X_norm))
        print(np.std(X_norm))
        feat_cols = ['feature' + str(i) for i in range(X_norm.shape[1])]
        normalised_dataset = pd.DataFrame(X_norm, columns=feat_cols)
        # print(normalised_dataset.tail())

        pca_result = PCA(n_components=n_components_1, svd_solver=svd_solver)
        principalComponents_pca_result = pca_result.fit_transform(X_norm)
        # print(principalComponents_pca_result)
        component_cols = ['principal_component_' + str(i) for i in range(n_components_1)]
        principalComponents_Df = pd.DataFrame(data=principalComponents_pca_result
                                           , columns=component_cols)
        principalComponents_Df[categorical_variable]=dataset[categorical_variable]
        # print(principalComponents_Df.tail())
        print('Explained variation per principal component: {}'.format(pca_result.explained_variance_ratio_))
        test_status = 'Unable to create scatter plot'
        max_axs=0
        for i in range(n_components_1):
            for j in range(i + 1, n_components_1):
                if i != j:
                    max_axs+=1
        # plt.figure(figsize=(10, 10))
        if max_axs<=1:max_axs=2
        fig, axs = plt.subplots(1, max_axs, figsize=(max_axs*8, 10),sharey='row')
        k=-1
        for i in range(n_components_1):
            for j in range(i+1, n_components_1):
                if i != j:
                    # print('i: {}, j: {}'.format(i,j))
                    # plt.figure()
                    # plt.figure(figsize=(10, 10))
                    # plt.xticks(fontsize=12)
                    # plt.yticks(fontsize=14)
                    k+=1
                    axs[k].set_xlabel('Principal Component - '+str(i), fontsize=20)
                    axs[k].set_ylabel('Principal Component - '+str(j), fontsize=20)
                    # axs[k].legend(loc='upper left')

                    axs[k].set_title("Principal Component Analysis", fontsize=20)
                    targets = pd.unique(dataset[categorical_variable])
                    colors = ["r", "g"] + ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(targets)-2)]

                    for target, color in zip(targets, colors):
                        indicesToKeep = dataset[categorical_variable] == target
                        axs[k].scatter(principalComponents_Df.loc[indicesToKeep, 'principal_component_'+str(i)]
                                    , principalComponents_Df.loc[indicesToKeep, 'principal_component_'+str(j)], c=color, s=50)
                        axs[k].legend(targets, prop={'size': 15})
        # plt.show()
        plt.tight_layout()
        if max_axs==2:
            fig.delaxes(axs[1])
        plt.savefig(path_to_storage + "/output/PCA.svg", format="svg")

        principal_axes = pd.DataFrame(pca_result.components_, columns=dataset.loc[:, independent_variables].columns,
                                      index=component_cols)
        principal_axes.insert(loc=0, column='Component', value=component_cols)
        explained_variance = pd.DataFrame(pca_result.explained_variance_, columns=['Variance'])
        explained_variance.insert(loc=0, column='Component', value=component_cols)
        explained_variance_ratio = pd.DataFrame()
        explained_variance_ratio = pd.DataFrame(pca_result.explained_variance_ratio_, columns=['Variance Ratio'])
        explained_variance_ratio.insert(loc=0, column='Component', value=component_cols)
        singular_values = pd.DataFrame(pca_result.singular_values_, columns=['Singular values'])
        singular_values.insert(loc=0, column='Component', value=component_cols)
        pd.DataFrame(data=principalComponents_pca_result,columns=component_cols).to_csv(path_to_storage + '/output/principalComponents_Df.csv', index=False)

        # list_1 = []
        # list_1.append(int(np.shape(X)[0]))
        # list_1.append(int(np.shape(X)[1]))
        # dim = min(list_1)
        # if n_components_2 == None:
        #     if n_components_1 > dim:
        #         return {'Error: n_components must be between 0 and min(n_samples, n_features)=': dim}
        #     pca = PCA(n_components=n_components_1)
        #     pca_t = pca.fit_transform(X)
        #     principal_Df = pd.DataFrame(data=pca_t, columns=["principalcomponent1", "principalcomponent2"])
        #     # principal_Df = pd.DataFrame(data=pca_t,
        #     #                             columns=['principal component 1', 'principal component 2'])
        #     print(principal_Df)
        #     print(dataset)
        #     pca.fit(X)
        # else:
        #     pca = PCA(n_components=n_components_2)
        #     pca.fit(X)

    # principal_Df = pd.DataFrame(data=pca, columns=['principal component 1', 'principal component 2'])
        test_status = 'Error in creating info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Principal component analysis',
                "test_params": {
                    "Independent variables": independent_variables
                },
                "test_results": {'columns': dataset.loc[:, independent_variables].columns.tolist(),
                                     'n_features_': pca_result.n_features_,
                                     'n_features_in_': pca_result.n_features_in_,
                                    'n_samples_': pca_result.n_samples_,
                                    'random_state': pca_result.random_state,
                                    'iterated_power': pca_result.iterated_power,
                                    'mean_': pca_result.mean_.tolist(),
                                    'explained_variance_': explained_variance.to_dict(),
                                    'noise_variance_': pca_result.noise_variance_,
                                    'pve': explained_variance_ratio.to_dict(),
                                    'singular_values': singular_values.to_dict(),
                                    'principal_axes': principal_axes.to_dict()}}
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/principalComponents_Df.csv'}
                                            ]
            file_data['Saved_plots'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                 step_id + '/PCA.svg'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()



        return JSONResponse(content={'status': 'Success',
                                     'columns': dataset.loc[:, independent_variables].columns.tolist(),
                                     'n_features_': pca_result.n_features_,
                                     'n_features_in_': pca_result.n_features_in_,
                                    'n_samples_': pca_result.n_samples_,
                                    'random_state': pca_result.random_state,
                                    'iterated_power': pca_result.iterated_power,
                                    'mean_': pca_result.mean_.tolist(),
                                    'explained_variance_': explained_variance.to_json(orient='records'),
                                    'noise_variance_': pca_result.noise_variance_,
                                    'pve': explained_variance_ratio.to_json(orient='records'),
                                    'singular_values': singular_values.to_json(orient='records'),
                                    'principal_axes': principal_axes.to_json(orient='records'),
                                     'principalComponents_Df':pd.DataFrame(data=principalComponents_pca_result
                                           , columns=component_cols).to_json(orient='records')},
                            status_code=200)

    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status,
                                     'columns': [],
                                     'n_features_': 0,
                                     'n_features_in_': 0,
                                     'n_samples_': 0,
                                     'random_state': 0,
                                     'iterated_power': 0,
                                     'mean_': [],
                                     'explained_variance_': [],
                                     'noise_variance_': 0,
                                     'pve': [],
                                     'singular_values': [],
                                     'principal_axes':[]
                                     },
                            status_code=200)
    # return {'Percentage of variance explained by each of the selected components': pca.explained_variance_ratio_.tolist(),
    #             'The singular values corresponding to each of the selected components. ': pca.singular_values_.tolist(),
    #             'Principal axes in feature space, representing the directions of maximum variance in the data.' : pca.components_.tolist()}

@router.get("/kmeans_clustering")
async def kmeans_clustering(workflow_id: str,
                            step_id: str,
                            run_id: str,
                            n_clusters: int,
                            independent_variables: list[str] | None = Query(default=None)):
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    to_return = {'cluster_centers': dfv.to_json(orient='records'), 'sum_squared_dist' : '','iterations_No': ''}
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        independent_variables = dfv['Variable'].tolist()
        selected_columns = pd.unique(dfv['Variable'])
        for columns in dataset.columns:
            if columns not in selected_columns:
                dataset = dataset.drop(str(columns), axis=1)
        test_status = 'Unable to compute KMeans. Variables with numeric values must be selected.'
        # X = np.array(dataset)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dataset)
        df = pd.DataFrame(kmeans.cluster_centers_, columns=dataset.columns)
        print(kmeans.cluster_centers_)
        dataset_new= dataset.insert(loc=0,column="Component", value=kmeans.labels_)
        print("dataset_new")
        print(dataset_new)
        print(kmeans.labels_)
        pd.DataFrame(data=dataset_new,columns=dataset.columns).to_csv(path_to_storage+'/output/new_dataset.csv',index=False)
        to_return={'cluster_centers': df.to_json(orient='records'), 'sum_squared_dist' : kmeans.inertia_,
                   'iterations_No': kmeans.n_iter_}
                # 'labels': kmeans.labels_.tolist(),
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": "KMeans",
                    "test_params": {
                        'selected_variables': independent_variables,
                        'n_clusters': n_clusters
                    },
                    "test_results": to_return
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/new_dataset.csv'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status':'Success', 'results': to_return}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'results': to_return}, status_code=200)


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
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Please provide all mandatory fields (dataset, dependent variable, one or more independent variables)'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        independent_variables = list(dfv['Variable'].values)
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        df_label = dataset[dependent_variable]

        for columns in dataset.columns:
            if columns not in independent_variables:
                dataset = dataset.drop(str(columns), axis=1)
        dataset.dropna(inplace=True)

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

        df_for_scatter = pd.concat(
            [pd.DataFrame(data={'Actual Values': list(Y), 'Predicted Values': list(clf.predict(X)),
                                'Residuals': list(Y - clf.predict(X))}), dataset], axis=1)
        values_dict = {}
        for column in df_for_scatter.columns:
            values_dict[column] = list(df_for_scatter[column])

        if np.shape(X)[1] == 1:
            coeffs = clf.coef_
            inter = clf.intercept_
            df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
            df_names = pd.DataFrame(dataset.columns, columns=['variables'])
            df = pd.concat([df_names, df_coeffs], axis=1)
        else:
            coeffs = np.squeeze(clf.coef_)
            inter = clf.intercept_
            df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
            df_names = pd.DataFrame(dataset.columns, columns=['variables'])
            df = pd.concat([df_names, df_coeffs], axis=1)
        response = {'skew': skew_res,
                    'kurtosis': kurt_res,
                    'Jarque Bera statistic': stat_jarq,
                    'Jarque-Bera p-value': p_jarq,
                    'Omnibus test statistic': omn_res_stat,
                    'Omnibus test p-value': omn_res_p,
                    'Durbin Watson': durb_res,
                    'actual_values': list(Y),
                    'predicted values': list(clf.predict(X)),
                    'residuals': list(Y - clf.predict(X)),
                    'coefficient of determination (R^2)': clf.score(X, Y),
                    'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                    'dataframe': df.to_html(), 'values_dict': values_dict,
                    'values_columns': list(df_for_scatter.columns),
                    'values_df': df_for_scatter.to_html()}

        df_for_scatter.to_csv(path_to_storage + '/output/elastic_preds.csv', index=False)
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'ElasticNet Regression',
                "test_params": {
                    'dependent variable': dependent_variable,
                    'independent variables': independent_variables,
                    'alpha': alpha,
                    'l1-ratio': l1_ratio,
                    'max iterations': max_iter
                },
                "test_results": {'skew': skew_res,
                                 'kurtosis': kurt_res,
                                 'Jarque Bera statistic': stat_jarq,
                                 'Jarque-Bera p-value': p_jarq,
                                 'Omnibus test statistic': omn_res_stat,
                                 'Omnibus test p-value': omn_res_p,
                                 'Durbin Watson': durb_res,
                                 'coefficient of determination (R^2)': clf.score(X, Y),
                                 'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                                 'dataframe': df.to_dict()}
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/analysis_output' + '/elastic_preds.csv'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Result': response},
                            status_code=200)
    except Exception as e:
        return JSONResponse(content={'status': test_status, 'Result': '[]'},
                            status_code=200)

# TODO Create frontend
@router.get("/lasso_regression")
async def lasso(workflow_id: str,
                step_id: str,
                run_id: str,
                dependent_variable: str,
                alpha: float | None = Query(default=1.0, gt=0),
                max_iter: int | None = Query(default=1000),
                independent_variables: list[str] | None = Query(default=None)):

    # dataset = load_file_csv_direct(workflow_id, run_id, step_id)



    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Please provide all mandatory fields (dataset, dependent variable, one or more independent variables)'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        independent_variables = list(dfv['Variable'].values)
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
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

        df_for_scatter = pd.concat(
            [pd.DataFrame(data={'Actual Values': list(Y), 'Predicted Values': list(clf.predict(X)),
                                'Residuals': list(Y - clf.predict(X))}), dataset], axis=1)
        values_dict = {}
        for column in df_for_scatter.columns:
            values_dict[column] = list(df_for_scatter[column])

        if np.shape(X)[1] == 1:
            coeffs = clf.coef_
            inter = clf.intercept_
            df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
            df_names = pd.DataFrame(dataset.columns, columns=['variables'])
            df = pd.concat([df_names, df_coeffs], axis=1)
        else:
            coeffs = np.squeeze(clf.coef_)
            inter = clf.intercept_
            df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
            df_names = pd.DataFrame(dataset.columns, columns=['variables'])
            df = pd.concat([df_names, df_coeffs], axis=1)
        response = {'skew': skew_res,
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
                'dataframe': df.to_json(orient='records'), 'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
                'values_df': df_for_scatter.to_json(orient='records')}


        df_for_scatter.to_csv(path_to_storage + '/output/lasso_preds.csv', index=False)
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Lasso Regression',
                "test_params": {
                    'dependent variable': dependent_variable,
                    'independent variables': independent_variables,
                    'alpha': alpha,
                    'max iterations': max_iter
                },
                "test_results": {'skew': skew_res,
                    'kurtosis': kurt_res,
                    'Jarque Bera statistic':stat_jarq,
                    'Jarque-Bera p-value': p_jarq,
                    'Omnibus test statistic': omn_res_stat,
                    'Omnibus test p-value': omn_res_p,
                    'Durbin Watson': durb_res,
                    'coefficient of determination (R^2)':clf.score(X,Y),
                    'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                    'dataframe': df.to_dict()}
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/lasso_preds.csv'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Result': response},
                            status_code=200)
    except Exception as e:
        return JSONResponse(content={'status': test_status, 'Result': '[]'},
                            status_code=200)

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
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Please provide all mandatory fields (dataset, dependent variable, one or more independent variables)'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        independent_variables = list(dfv['Variable'].values)
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        df_label = dataset[dependent_variable]

        for columns in dataset.columns:
            if columns not in independent_variables:
                dataset = dataset.drop(str(columns), axis=1)

        ## probably that instead of data?
        dataset.dropna(inplace=True)

        X = np.array(dataset)
        Y = np.array(df_label.astype('float64'))

        if solver != 'lbfgs':
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

        df_for_scatter = pd.concat(
            [pd.DataFrame(data={'Actual Values': list(Y), 'Predicted Values': list(clf.predict(X)),
                                'Residuals': list(Y - clf.predict(X))}), dataset], axis=1)
        values_dict = {}
        for column in df_for_scatter.columns:
            values_dict[column] = list(df_for_scatter[column])

        if np.shape(X)[1] == 1:
            coeffs = clf.coef_
            inter = clf.intercept_
            df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
            df_names = pd.DataFrame(dataset.columns, columns=['variables'])
            df = pd.concat([df_names, df_coeffs], axis=1)
        else:
            coeffs = np.squeeze(clf.coef_)
            inter = clf.intercept_
            df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
            df_names = pd.DataFrame(dataset.columns, columns=['variables'])
            df = pd.concat([df_names, df_coeffs], axis=1)
        response = {'skew': skew_res,
                    'kurtosis': kurt_res,
                    'Jarque Bera statistic': stat_jarq,
                    'Jarque-Bera p-value': p_jarq,
                    'Omnibus test statistic': omn_res_stat,
                    'Omnibus test p-value': omn_res_p,
                    'Durbin Watson': durb_res,
                    'actual_values': list(Y),
                    'predicted values': list(clf.predict(X)),
                    'residuals': list(Y - clf.predict(X)),
                    'coefficient of determination (R^2)': clf.score(X, Y),
                    'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                    'dataframe': df.to_html(), 'values_dict': values_dict,
                    'values_columns': list(df_for_scatter.columns),
                    'values_df': df_for_scatter.to_html()}

        df_for_scatter.to_csv(path_to_storage + '/output/ridge_preds.csv', index=False)
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Ridge Regression',
                "test_params": {
                    'dependent variable': dependent_variable,
                    'independent variables': independent_variables,
                    'alpha': alpha,
                    'solver': solver,
                    'max iterations': max_iter
                },
                "test_results": {'skew': skew_res,
                                 'kurtosis': kurt_res,
                                 'Jarque Bera statistic': stat_jarq,
                                 'Jarque-Bera p-value': p_jarq,
                                 'Omnibus test statistic': omn_res_stat,
                                 'Omnibus test p-value': omn_res_p,
                                 'Durbin Watson': durb_res,
                                 'coefficient of determination (R^2)': clf.score(X, Y),
                                 'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                                 'dataframe': df.to_dict()}
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/ridge_preds.csv'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Result': response},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'Result': '[]'},
                            status_code=200)

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
    print("loukas")
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
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Please provide all mandatory fields (dataset, dependent variable, one or more independent variables)'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        independent_variables = list(dfv['Variable'].values)
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        df_label = dataset[dependent_variable]

        for columns in dataset.columns:
            if columns not in independent_variables:
                dataset = dataset.drop(str(columns), axis=1)
        dataset.dropna(inplace=True)

        X = np.array(dataset)
        Y = np.array(df_label.astype('float64'))

        if loss == 'huber' or loss == 'epsilon_insensitive' or loss == 'squared_epsilon_insensitive':
            if learning_rate == 'constant' or learning_rate == 'invscaling' or learning_rate == 'adaptive':
                clf = SGDRegressor(alpha=alpha, max_iter=max_iter, epsilon=epsilon, eta0=eta0, penalty=penalty,
                                   l1_ratio=l1_ratio, learning_rate=learning_rate)
            else:
                clf = SGDRegressor(alpha=alpha, max_iter=max_iter, epsilon=epsilon, penalty=penalty, l1_ratio=l1_ratio,
                                   learning_rate=learning_rate)
        else:
            clf = SGDRegressor(alpha=alpha, max_iter=max_iter, eta0=eta0, penalty=penalty, l1_ratio=l1_ratio,
                               learning_rate=learning_rate)

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

        else:
            coeffs = np.squeeze(clf.coef_)

        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        response = {'skew': skew_res,
                    'kurtosis': kurt_res,
                    'Jarque Bera statistic': stat_jarq,
                    'Jarque-Bera p-value': p_jarq,
                    'Omnibus test statistic': omn_res_stat,
                    'Omnibus test p-value': omn_res_p,
                    'Durbin Watson': durb_res,
                    'actual_values': list(Y),
                    'predicted values': list(clf.predict(X)),
                    'residuals': list(Y - clf.predict(X)),
                    'coefficient of determination (R^2)': clf.score(X, Y),
                    'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_html(),
                    'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
                    'values_df': df_for_scatter.to_html()}

        df_for_scatter.to_csv(path_to_storage + '/output/sgd_preds.csv', index=False)
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'SGD Regression',
                "test_params": {
                    'dependent variable': dependent_variable,
                    'independent variables': independent_variables,
                    'alpha': alpha,
                    'max iterations': max_iter,
                    'epsilon': epsilon,
                    'eta0': eta0,
                    'l1_ratio': l1_ratio,
                    'loss': loss,
                    'learning rate': learning_rate,
                    'penalty': penalty
                },
                "test_results": {'skew': skew_res,
                                 'kurtosis': kurt_res,
                                 'Jarque Bera statistic': stat_jarq,
                                 'Jarque-Bera p-value': p_jarq,
                                 'Omnibus test statistic': omn_res_stat,
                                 'Omnibus test p-value': omn_res_p,
                                 'Durbin Watson': durb_res,
                                 'coefficient of determination (R^2)': clf.score(X, Y),
                                 'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                                 'dataframe': df.to_dict()}
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/sgd_preds.csv'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Result': response},
                            status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        return JSONResponse(content={'status': test_status, 'Result': '[]'},
                            status_code=200)

@router.get("/huber_regression")
async def huber_regressor(workflow_id: str,
                        run_id: str,
                        step_id: str,
                        dependent_variable: str,
                        alpha: float | None = Query(default=0.0001),
                        max_iter: int | None = Query(default=100),
                        epsilon: float | None = Query(default=1.35),
                          independent_variables: list[str] | None = Query(default=None)):
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Please provide all mandatory fields (dataset, dependent variable, one or more independent variables)'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        independent_variables = list(dfv['Variable'].values)
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        df_label = dataset[dependent_variable]

        for columns in dataset.columns:
            if columns not in independent_variables:
                dataset = dataset.drop(str(columns), axis=1)
        dataset.dropna(inplace=True)

        X = np.array(dataset)
        Y = np.array(df_label.astype('float64'))
        
        if epsilon < 1:
            test_status = 'Epsilon must take float values equal or higher to 1'
            raise Exception

        if alpha < 0:
            test_status = 'Alpha must take float values equal or higher to 0'
            raise Exception

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

        else:
            coeffs = np.squeeze(clf.coef_)

        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        response = {'skew': skew_res,
                    'kurtosis': kurt_res,
                    'Jarque Bera statistic': stat_jarq,
                    'Jarque-Bera p-value': p_jarq,
                    'Omnibus test statistic': omn_res_stat,
                    'Omnibus test p-value': omn_res_p,
                    'Durbin Watson': durb_res,
                    'actual_values': list(Y),
                    'predicted values': list(clf.predict(X)),
                    'residuals': list(Y - clf.predict(X)),
                    'coefficient of determination (R^2)': clf.score(X, Y),
                    'coefficients': coeffs.tolist(), 'intercept': inter.tolist(), 'dataframe': df.to_html(),
                    'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
                    'values_df': df_for_scatter.to_html()}

        df_for_scatter.to_csv(path_to_storage + '/output/huber_preds.csv', index=False)
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Huber Regression',
                "test_params": {
                    'dependent variable': dependent_variable,
                    'independent variables': independent_variables,
                    'alpha': alpha,
                    'max iterations': max_iter,
                    'epsilon': epsilon
                },
                "test_results": {'skew': skew_res,
                                 'kurtosis': kurt_res,
                                 'Jarque Bera statistic': stat_jarq,
                                 'Jarque-Bera p-value': p_jarq,
                                 'Omnibus test statistic': omn_res_stat,
                                 'Omnibus test p-value': omn_res_p,
                                 'Durbin Watson': durb_res,
                                 'coefficient of determination (R^2)': clf.score(X, Y),
                                 'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                                 'dataframe': df.to_dict()}
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/huber_preds.csv'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Result': response},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'Result': '[]'},
                            status_code=200)

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

    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Please provide all mandatory fields (dataset, dependent variable, one or more independent variables)'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        independent_variables = list(dfv['Variable'].values)
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        df_label = dataset[dependent_variable]

        for columns in dataset.columns:
            if columns not in independent_variables:
                dataset = dataset.drop(str(columns), axis=1)
        dataset.dropna(inplace=True)

        X = np.array(dataset)
        Y = np.array(df_label.astype('float64'))

        if epsilon < 0:
            test_status = 'Epsilon must take float values equal or higher to 0'
            raise Exception

        if C <= 0:
            test_status = 'C must take positive values'
            raise Exception

        clf = LinearSVR(loss=loss, C=C, epsilon=epsilon, max_iter=max_iter)

        clf.fit(X, Y)

        residuals = Y - clf.predict(X)
        skew_res = skew(residuals)
        kurt_res = kurtosis(residuals)
        jarq_res = jarque_bera(residuals)
        stat_jarq = jarq_res.statistic
        p_jarq = jarq_res.pvalue
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
        response = {'skew': skew_res,
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

        df_for_scatter.to_csv(path_to_storage + '/output/linearsvr_preds.csv', index=False)
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'LinearSVR',
                "test_params": {
                    'dependent variable': dependent_variable,
                    'independent variables': independent_variables,
                    'C': C,
                    'loss': loss,
                    'max iterations': max_iter,
                    'epsilon': epsilon
                },
                "test_results": {'skew': skew_res,
                                 'kurtosis': kurt_res,
                                 'Jarque Bera statistic': stat_jarq,
                                 'Jarque-Bera p-value': p_jarq,
                                 'Omnibus test statistic': omn_res_stat,
                                 'Omnibus test p-value': omn_res_p,
                                 'Durbin Watson': durb_res,
                                 'coefficient of determination (R^2)': clf.score(X, Y),
                                 'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                                 'dataframe': df.to_dict()}
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/linearsvr_preds.csv'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Result': response},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'Result': '[]'},
                            status_code=200)



@router.get("/linearsvc_regression")
async def linear_svc_regressor(workflow_id: str,
                               step_id: str,
                               run_id: str,
                               dependent_variable: str,
                               max_iter: int | None = Query(default=1000),
                               C: float | None = Query(default=1,gt=0),
                               loss: str | None = Query("squared_hinge",
                                                         regex="^(hinge)$|^(squared_hinge)$"),
                               penalty: str | None = Query("l2",
                                                         regex="^(l1)$|^(l2)$"),
                               independent_variables: list[str] | None = Query(default=None)):

    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Please provide all mandatory fields (dataset, dependent variable, one or more independent variables)'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        independent_variables = list(dfv['Variable'].values)
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        df_label = dataset[dependent_variable]

        for columns in dataset.columns:
            if columns not in independent_variables:
                dataset = dataset.drop(str(columns), axis=1)

        features_columns = dataset.columns

        X = np.array(dataset)
        Y = np.array(df_label.astype('float64'))

        if C <= 0:
            test_status = 'C must take positive values'
            raise Exception

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

        response = {'coefficients': df_coefs.to_json(orient='records'), 'intercept': df_intercept.to_json(orient='records'), 'values_dict': values_dict,
            'values_columns': list(df_for_scatter.columns),
            'values_df': df_for_scatter.to_json(orient='records')}

        df_for_scatter.to_csv(path_to_storage + '/output/linearsvc.csv', index=False)
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'LinearSVC',
                "test_params": {
                    'dependent variable': dependent_variable,
                    'independent variables': independent_variables,
                    'C': C,
                    'loss': loss,
                    'max iterations': max_iter,
                    'penalty': penalty
                },
                "test_results": {}
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/linearsvc.csv'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Result': response},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'Result': '[]'},
                            status_code=200)



@router.get("/ancova")
async def ancova_2(workflow_id: str,
                    step_id: str,
                    run_id: str,
                   dv: str,
                   between: str,
                   covar: list[str] | None = Query(default=None),
                   effsize: str | None = Query("np2",
                                               regex="^(np2)$|^(n2)$")):
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dv = dv.split("--")[1]
        between = between.split("--")[1]
        dfv['variables'] = covar
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        df_data = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        covar = dfv['Variable'].tolist()

        test_status = 'Unable to compute Ancova test for the selected columns. Nonnumeric values are selected for the Dependent variable or the Covariates.'
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
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Ancova test',
                "test_params": {
                    'selected_depedent_variable': dv,
                    'selected_between_factor':between,
                    'selected_covariate_variables':covar
                },
                "test_results": all_res,
                "Output_datasets":[],
                'Saved_plots': []
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        print(all_res)
        return JSONResponse(content={'status': 'Success','DataFrame': all_res},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'DataFrame': []},
                            status_code=200)
    # return {'ANCOVA':df.to_json(orient="split")}

@router.get("/linear_mixed_effects_model")
async def linear_mixed_effects_model(workflow_id: str,
                    step_id: str,
                    run_id: str,
                     dependent: str,
                     groups: str,
                     independent: list[str] | None = Query(default=None),
                     use_sqrt: bool | None = Query(default=True)):
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dependent = dependent.split("--")[1]
        groups = groups.split("--")[1]
        dfv['variables'] = independent
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        data = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        independent = dfv['Variable'].tolist()

        test_status = 'Unable to compute Mixed Linear Model Regression test for the selected columns. Nonnumeric values are selected for the Dependent variable.'
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

        test_status = 'Erro in creating info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Linear Mixed Effects Model',
                "test_params": {
                    'selected_depedent_variable': dependent,
                    'selected_groups':groups,
                    'selected_covariate_variables':independent
                },
                "test_results": {
                    'model':tbl1_res,
                    'coeficients':tbl2_res
                },
                "Output_datasets":[],
                'Saved_plots': []
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success','first_table': tbl1_res, 'second_table': tbl2_res},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'first_table': [], 'second_table': []},
                            status_code=200)
    # return {'first_table': df_0.to_json(orient='split'), 'second_table': df_1.to_json(orient='split')}

@router.get("/poisson_regression")
async def poisson_regression(workflow_id: str,
                             step_id: str,
                             run_id: str,
                             dependent_variable: str,
                             alpha: float | None = Query(default=1.0, ge=0),
                             max_iter: int | None = Query(default=1000),
                             independent_variables: list[str] | None = Query(default=None)):
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Please provide all mandatory fields (dataset, dependent variable, one or more independent variables)'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        independent_variables = list(dfv['Variable'].values)
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        df_label = dataset[dependent_variable]

        for columns in dataset.columns:
            if columns not in independent_variables:
                dataset = dataset.drop(str(columns), axis=1)

        X = np.array(dataset)
        Y = np.array(df_label.astype('float64'))

        test_status = 'Cannot run the algorithm in this configuration'
        clf = PoissonRegressor(alpha=alpha, max_iter=max_iter)

        X = sklearn.preprocessing.MinMaxScaler().fit_transform(X)

        clf.fit(X, Y)

        residuals = Y - clf.predict(X)
        skew_res = skew(residuals)
        kurt_res = kurtosis(residuals)
        jarq_res = jarque_bera(residuals)
        stat_jarq = jarq_res.statistic
        p_jarq = jarq_res.pvalue
        omn_res_stat, omn_res_p = normaltest(residuals)
        durb_res = durbin_watson(residuals)

        df_for_scatter = pd.concat(
            [pd.DataFrame(data={'Actual Values': list(Y), 'Predicted Values': list(clf.predict(X)),
                                'Residuals': list(Y - clf.predict(X))}), dataset], axis=1)
        values_dict = {}
        for column in df_for_scatter.columns:
            values_dict[column] = list(df_for_scatter[column])

        if np.shape(X)[1] == 1:
            coeffs = clf.coef_
        else:
            coeffs = np.squeeze(clf.coef_)
        inter = clf.intercept_
        df_coeffs = pd.DataFrame(coeffs, columns=['coefficients'])
        df_names = pd.DataFrame(dataset.columns, columns=['variables'])
        df = pd.concat([df_names, df_coeffs], axis=1)
        response = {'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                'dataframe': df.to_json(orient='records'), 'df_for_scatter': df_for_scatter.to_json(orient='records'),
                    'values_dict': values_dict, 'values_columns': list(df_for_scatter.columns),
                    'skew': skew_res,
                    'kurtosis': kurt_res,
                    'Jarque Bera statistic': stat_jarq,
                    'Jarque-Bera p-value': p_jarq,
                    'Omnibus test statistic': omn_res_stat,
                    'Omnibus test p-value': omn_res_p,
                    'Durbin Watson': durb_res,
                    'actual_values': list(Y),
                    'predicted values': list(clf.predict(X)),
                    'residuals': list(Y - clf.predict(X)),
                    'coefficient of determination (R^2)': clf.score(X, Y)}
        df_for_scatter.to_csv(path_to_storage + '/output/poisson_preds.csv', index=False)
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Poisson Regression',
                "test_params": {
                    'dependent variable': dependent_variable,
                    'independent variables': independent_variables,
                    'alpha': alpha,
                    'max iterations': max_iter
                },
                "test_results": {'skew': skew_res,
                    'kurtosis': kurt_res,
                    'Jarque Bera statistic':stat_jarq,
                    'Jarque-Bera p-value': p_jarq,
                    'Omnibus test statistic': omn_res_stat,
                    'Omnibus test p-value': omn_res_p,
                    'Durbin Watson': durb_res,
                    'coefficient of determination (R^2)':clf.score(X,Y),'coefficients': coeffs.tolist(), 'intercept': inter.tolist(),
                                 'dataframe': df.to_dict()}
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/poisson_preds.csv'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Result': response},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'Result': '[]'},
                            status_code=200)

@router.get("/cox_regression")
async def cox_regression(workflow_id: str,
                         step_id: str,
                         run_id: str,
                         duration_col: str,
                         covariates: list[str] | None = Query(default=None),
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
                         hazard_ratios: bool | None = Query(default=False),
                         baseline_estimation_method: str | None = Query("breslow",
                                                                       regex="^(breslow)$|^(spline)$|^(piecewise)$")):

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    dfv = pd.DataFrame()
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        #TODO mandatory fields add
        test_status = 'Please provide all mandatory fields (dataset, dependent variable, one or more independent variables)'
        dfv['covariates'] = covariates
        dfv[['Datasource', 'covariates']] = dfv["covariates"].apply(lambda x: pd.Series(str(x).split("--")))
        covariates = list(dfv['covariates'].values)
        selected_datasource = pd.unique(dfv['Datasource'])[0]
        try:
            strata = list(map(lambda x: str(x).split("--")[1], strata))
        except:
            strata=None

        test_status = 'Unable to retrieve datasets'
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasource)
        test_status = 'Unable to compute generalized estimating equations for the selected columns.'

        to_return = []

        fig = plt.figure(1)
        ax = plt.subplot(111)

        # dataset = pd.read_csv('example_data/mescobrad_dataset.csv')

        if baseline_estimation_method == "spline":
            cph = CoxPHFitter(alpha=alpha, baseline_estimation_method=baseline_estimation_method, penalizer=penalizer,
                              l1_ratio=l1_ratio, strata=strata,
                              n_baseline_knots=n_baseline_knots)
        elif baseline_estimation_method == "piecewise":
            cph = CoxPHFitter(alpha=alpha, baseline_estimation_method=baseline_estimation_method, penalizer=penalizer,
                              l1_ratio=l1_ratio, strata=strata,
                              breakpoints=breakpoints)
        else:
            cph = CoxPHFitter(alpha=alpha, baseline_estimation_method=baseline_estimation_method, penalizer=penalizer,
                              l1_ratio=l1_ratio, strata=strata)
        if weights_col=='': weights_col=None
        cph.fit(dataset, duration_col=duration_col, event_col=event_col, weights_col=weights_col)
                #cluster_col=cluster_col, entry_col=entry_col)

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
        # fig = plt.figure(figsize=(18, 12))
        cph.plot(hazard_ratios=hazard_ratios)
        html_str = mpld3.fig_to_html(fig)
        to_return.append({"figure_1": html_str})
        plt.clf()
        if covariates != None:
            fig = plt.figure(1)
            values = []
            for covariate in covariates:
                values.append(list(pd.unique(dataset[covariate])))
            values = list(itertools.product(*values))
            ax = cph.plot_partial_effects_on_outcome(covariates=covariates, values=values, cmap='coolwarm')
            ax.plot()
            html_str = mpld3.fig_to_html(ax.get_figure())
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

        test_status = 'Error in creating info file.'
        # with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
        #     file_data = json.load(f)
        #     file_data['results'] |= {
        #         "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        #         "workflow_id": workflow_id,
        #         "run_id": run_id,
        #         "step_id": step_id,
        #         "test_name": 'Generalized Estimating Equations',
        #         "test_params": {
        #             "dependent_variable": dependent_variable,
        #             "groups": groups,
        #             "independent_variables": independent_variables,
        #             "cov_struct": cov_struct,
        #             "family": family
        #         },
        #         "test_results": {
        #             "first_table": df_0.to_dict(),
        #             "second_table": df_1.to_dict(),
        #             "third_table": df_2.to_dict(),
        #         },
        #         "Output_datasets": [],
        #         'Saved_plots': []
        #     }
        #     f.seek(0)
        #     json.dump(file_data, f, indent=4)
        #     f.truncate()
        return JSONResponse(content={'status': 'Success',
                                     'Concordance_Index':cph.concordance_index_,
                                     'AIC': AIC,
                                     'Dataframe': tbl1_res,
                                     'figure': to_return,
                                     'proportional_hazard_test': tbl2_res},
                            status_code=200)
    except Exception as e:
        return JSONResponse(content={'status': test_status,
                                     'Concordance_Index':[],
                                     'AIC': [],
                                     'Dataframe': [],
                                     'figure': [],
                                     'proportional_hazard_test': []},
                            status_code=200)

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
    print(cph)
    cph.fit(dataset_long, event_col=event_col, id_col='id', weights_col=weights_col,start_col='start', stop_col='stop', strata=strata)

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

@router.get("/anova_pairwise_tests")
async def anova_pairwise_tests(workflow_id: str,
                               run_id: str,
                               step_id: str,
                               dv: str,
                               subject: str,
                               marginal: str,
                               alpha: str,
                               alternative: str,
                               padjust: str,
                               effsize: str,
                               correction: str,
                               nan_policy: str,
                               between: list[str] | None = Query(default=None),
                               within: list[str] | None = Query(default=None),
                               ):
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        selected_datasource = dv.split("--")[0]
        #TODO CHANGE THIS ON SIMPLE ANOVA!!!
        dependent_variable = dv.split("--")[1]
        dependent_variable_ping = re.sub("[\(\),:]", " ", dependent_variable)

        subject = subject.split("--")[1]
        subject_ping = re.sub("[\(\),:]", " ", subject)

        test_status = 'Wrong variables'

        between_factor = list(map(lambda x: str(x).split("--")[1], between))
        between_factor_ping = list(map(lambda x: re.sub("[\(\),:]", " ", x), between_factor))
        if between_factor_ping == ["None"]:
            between_factor_ping = None

        within_factor = list(map(lambda x: str(x).split("--")[1], within))
        within_factor_ping = list(map(lambda x: re.sub("[\(\),:]", " ", x), within_factor))
        if within_factor_ping == ["None"]:
            within_factor_ping = None


        alpha = float(alpha)
        marginal = (marginal == "True")

        if not (correction == "auto"):
            correction = (correction == "True")

        test_status = 'within or between must have a value'
        assert not (within_factor_ping == None and between_factor_ping == None)

        test_status = 'A column can not be selected multiple times'
        var_list = [dependent_variable] + within_factor + between_factor + [subject]
        assert len(var_list) == len(set(var_list))

        test_status = 'Unable to retrieve datasets'
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasource)
        pd.set_option('display.max_columns', None)
        dataset.columns = list(map(lambda x: re.sub("[\(\),:]", " ", x), dataset.columns))

        df = pingouin.pairwise_tests(data=dataset, dv=dependent_variable_ping, within=within_factor_ping, between=between_factor_ping,
                                     subject=subject_ping, marginal=marginal, alpha=alpha, alternative=alternative,
                                     padjust=padjust, effsize=effsize, correction=correction, nan_policy=nan_policy)
        df = df.fillna('')

        columns = [{
            "col" : "id"}]

        for col in df.columns:
            match col:
                case "A":
                    new_col = "1st measurement"
                case "B":
                    new_col = "2nd measurement"
                case "dof":
                    new_col = "Deg. of Fr."
                case "p-unc":
                    new_col = "p-uncorrected"
                case "p-corr":
                    new_col = "p-corrected"
                case "p-adjust":
                    new_col = "Correction method"
                case "BF10":
                    new_col = "Bayes Factor"
                case _:
                    new_col = col
            columns.append({
                "col": new_col
            })
            df.rename(columns={col: new_col}, inplace=True)

        print(columns)

        all_res = []
        for ind, row in df.iterrows():
            temp_to_append = row.to_dict()
            temp_to_append['id'] = ind
            all_res.append(temp_to_append)
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Pairwise Tests',
                "test_params": {
                    'selected_depedent_variable': dependent_variable,
                    'selected_subject': subject,
                    'selected_marginal': marginal,
                    'selected_alpha': alpha,
                    'selected_alternative': alternative,
                    'selected_padjust': padjust,
                    'selected_effsize': effsize,
                    'selected_correction': correction,
                    'selected_nan_policy': nan_policy,
                    'selected_between': between_factor,
                    'selected_within': within_factor,
                },
                "test_results": all_res,
                "Output_datasets": [],
                'Saved_plots': []
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        print(all_res)


        return JSONResponse(content={'status': 'Success', 'DataFrame': all_res, 'Columns': columns},
                            status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={'status': test_status, 'DataFrame': [], 'Columns': []},
                            status_code=200)
@router.get("/anova_repeated_measures")
async def anova_rm(workflow_id: str,
                   step_id: str,
                   run_id: str,
                   dependent_variable: str,
                   subject: str,
                   within: list[str] | None = Query(default=None),
                   aggregate_func: str | None = Query(default=None,
                                                      regex="^(mean)$")):

    df_data = pd.read_csv('C:\\neurodesktop-storage\\runtime_config\\workflow_3fa85f64-5717-4562-b3fc-2c963f66afa6\\run_3fa85f64-5717-4562-b3fc-2c963f66afa6\\step_3fa85f64-5717-4562-b3fc-2c963f66afa6/Sample_rep_measures.csv')
    # df_data = load_file_csv_direct(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    print(df_data.columns)
    # unique, counts = np.unique(df_data[subject], return_counts=True)
    # print(unique)
    # print(counts)
    # z = all(x==counts[0] for x in counts)
    # print(z)
    print(dependent_variable)
    print(subject)
    print(within)
    print(df_data)
    # posthocs = pingouin.pairwise_ttests(dv=dependent_variable,
    #                                     within=within, between='Age',
    #                                     subject=subject, data=df_data)
    # pingouin.print_table(posthocs)

    z=True
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

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    dfv = pd.DataFrame()
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        test_status = 'Please provide all mandatory fields (dataset, dependent variable, one or more independent variables)'
        z = dependent_variable + " ~ "
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        independent_variables = list(dfv['Variable'].values)
        selected_datasource = pd.unique(dfv['Datasource'])[0]
        z = z + " + ".join(independent_variables)

        test_status = 'Unable to retrieve datasets'
        data = load_data_from_csv(path_to_storage + "/" + selected_datasource)
        test_status = 'Unable to compute generalized estimating equations for the selected columns.'

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

        md = smf.gee(formula=z, groups=groups, data=data, cov_struct=ind, family=fam)

        mdf = md.fit()

        #print(md.predict())
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

        df_1.to_csv(path_to_storage + '/output/generalized_estimating_equations.csv', index=False)
        results_as_html = df.tables[2].as_html()
        df_2 = pd.read_html(results_as_html)[0]
        df_new = df_2[[2, 3]]
        df_2.drop(columns=[2, 3], inplace=True)
        df_2 = pd.concat([df_2, df_new.rename(columns={2: 0, 3: 1})], ignore_index=True)
        df_2.set_index(0, inplace=True)
        df_2.index.name = None
        df_2.rename(columns={1: 'Values'}, inplace=True)
        df_2.reset_index(inplace=True)
        test_status = 'Error in creating info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Generalized Estimating Equations',
                "test_params": {
                    "dependent_variable" : dependent_variable,
                    "groups": groups,
                    "independent_variables" : independent_variables,
                    "cov_struct" : cov_struct,
                    "family" : family
                },
                "test_results": {
                    "first_table":df_0.to_dict(),
                    "second_table":df_1.to_dict(),
                    "third_table":df_2.to_dict(),
                },
                "Output_datasets": [],
                'Saved_plots': []
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'first_table':df_0.to_json(orient='records'),
                                     'second_table':df_1.to_json(orient='records'),
                                     'third_table':df_2.to_json(orient='records')},
                            status_code=200)
    except Exception as e:
        return JSONResponse(content={'status': test_status,
                                     'first_table':[],
                                     'second_table':[],
                                     'third_table':[]},
                                status_code=200)


@router.get("/kaplan_meier")
async def kaplan_meier(workflow_id: str,
                       step_id: str,
                       run_id: str,
                       column_1: str,
                       column_2: str,
                       at_risk_counts: bool | None = Query(default=True),
                       label: str | None = Query(default=None),
                       alpha: float | None = Query(default=0.05)):
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        selected_datasource = column_1.split("--")[0]
        column_1 = column_1.split("--")[1]
        column_2 = column_2.split("--")[1]
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasource)

        test_status = 'Unable to compute Kaplan Meier Fitter test for the selected columns. '
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
        test_status = 'Erro in creating info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Kaplan Meier Fitter',
                "test_params": {
                    'selected_exposure_variable': column_1,
                    'selected_outcome_variable': column_2,
                    'selected_at_risk_counts': at_risk_counts,
                    'selected_alpha': alpha,
                    'selected_label': label,
                },
                "test_results": {
                    "survival_function":df.to_dict(),
                    "confidence_interval": confidence_interval.to_dict(),
                    'event_table': event_table.to_dict(),
                    "conditional_time_to_event": conditional_time_to_event.to_dict(),
                    "confidence_interval_cumulative_density":confidence_interval_cumulative_density.to_dict(),
                    "cumulative_density" : cumulative_density.to_dict(),
                    "timeline" : timeline.to_dict(),
                    "median_survival_time": str(median_survival_time)
                },
                "Output_datasets": [],
                'Saved_plots': [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                             step_id + '/survival_function.svg'}
                                    ]
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', "survival_function":df.to_json(orient="records"),
                                     "confidence_interval": confidence_interval.to_json(orient='records'),
                                     'event_table': event_table.to_json(orient="records"),
                                     "conditional_time_to_event": conditional_time_to_event.to_json(orient="records"),
                                     "confidence_interval_cumulative_density":confidence_interval_cumulative_density.to_json(orient="records"),
                                     "cumulative_density" : cumulative_density.to_json(orient='records'),
                                     "timeline" : timeline.to_json(orient='records'),
                                     "median_survival_time": str(median_survival_time)},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, "survival_function":'[]',
                                         "confidence_interval": '[]',
                                         'event_table': '[]',
                                         "conditional_time_to_event": '[]',
                                         "confidence_interval_cumulative_density":'[]',
                                         "cumulative_density" : '[]',
                                         "timeline" : '[]',
                                         "median_survival_time": ''},
                                status_code=200)


@router.get("/fisher")
async def fisher(
        workflow_id: str,
        step_id: str,
        run_id: str,
        variable_column: str,
        variable_row: str,
        alternative: Optional[str] | None = Query("two-sided",
                                                  regex="^(two-sided)$|^(less)$|^(greater)$")):
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        selected_datasource = variable_column.split("--")[0]
        variable_column = variable_column.split("--")[1]
        variable_row = variable_row.split("--")[1]
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        data = load_data_from_csv(path_to_storage + "/" + selected_datasource)

        test_status = 'Unable to compute Fisher exact test for the selected columns. '
        row_var = data[variable_row]
        column_var = data[variable_column]

        df = pd.crosstab(index=row_var,columns=column_var)
        df1 = pd.crosstab(index=row_var,columns=column_var, margins=True, margins_name= "Total")
        odd_ratio, p_value = fisher_exact(df, alternative=alternative)

        test_status = 'Error in creating info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Fisher exact',
                "test_params": {
                    'variable_column': variable_column,
                    'variable_row': variable_row,
                    'alternative': alternative
                },
                "test_results": {
                    'odd_ratio': odd_ratio,
                    "p_value": p_value,
                    "crosstab":df1.to_dict()
                },
                "Output_datasets": [],
                'Saved_plots': []
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'odd_ratio': odd_ratio, "p_value": p_value, "crosstab":df1.to_json(orient='split')},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'odd_ratio': '', "p_value": '', "crosstab":"{\"columns\":[0,1,\"Total\"],\"index\":[0,1,\"Total\"],\"data\":[[0,0,0],[0,0,0],[0,0,0]]}"},
                            status_code=200)

@router.get("/mc_nemar")
async def mc_nemar(workflow_id: str,
                   step_id: str,
                   run_id: str,
                   variable_column: str,
                   variable_row: str,
                   exact: bool | None = Query(default=False),
                   correction: bool | None = Query(default=True)):
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        selected_datasource = variable_column.split("--")[0]
        variable_column = variable_column.split("--")[1]
        variable_row = variable_row.split("--")[1]
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        data = load_data_from_csv(path_to_storage + "/" + selected_datasource)
        test_status = 'Unable to compute McNemar test for the selected columns.'

        # I used LabelEncoder() to convert str to int, but I made it a comment
        # because the user does not have the tranformed
        # dataset to know what 0 and 1 means
        df_tranf = pd.DataFrame()
        row_var = data[variable_row]
        if row_var.dtypes != 'int64':
            le = LabelEncoder()
            row_var = le.fit_transform(data[variable_row])
            df_tranf['index'] = [0, 1]
            df_tranf[variable_row] = [str(x) for x in le.classes_]
        column_var = data[variable_column]
        if column_var.dtypes != 'int64':
            le = LabelEncoder()
            column_var = le.fit_transform(data[variable_column])
            df_tranf['index'] = [0, 1]
            df_tranf[variable_column] = [str(x) for x in le.classes_]

        df = pd.crosstab(index=row_var,columns=column_var)
        df1 = pd.crosstab(index=row_var,columns=column_var, margins=True, margins_name= "Total")

        result = mcnemar(df, exact=exact, correction=correction)
        test_status = 'Error in creating info file.'
        statistic = result.statistic if not np.isinf(result.statistic) else 'infinity'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'McNemar',
                "test_params": {
                    'variable_column': variable_column,
                    'variable_row': variable_row,
                    'exact': exact,
                    'correction': correction
                },
                "test_results": {
                    'statistic': statistic,
                    "p_value": result.pvalue,
                    "crosstab": df1.to_dict()
                },
                "Output_datasets": [],
                'Saved_plots': []
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'statistic': result.statistic,
                                     "p_value": result.pvalue,
                                     "crosstab":df1.to_json(orient='split'), 'col_transormed':df_tranf.to_json(orient="records")},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'statistic': '', "p_value": '',
                                     "crosstab": "{\"columns\":[0,1,\"Total\"],\"index\":[0,1,\"Total\"],\"data\":[[0,0,0],[0,0,0],[0,0,0]]}", 'col_transormed':'[]'},
                            status_code=200)
    # return {'statistic': result.statistic, "p_value": result.pvalue, "crosstab":df1.to_json(orient='split')}

# TODO: Do we need this?
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
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        selected_datasource = exposure.split("--")[0]
        exposure = exposure.split("--")[1]
        outcome = outcome.split("--")[1]
        if time is not None: time = time.split("--")[1]
        else: time = None
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasource)
        # Change binary str values to 0,1
        df_tranf = pd.DataFrame()
        if dataset[exposure].dtypes != 'int64':
            le = LabelEncoder()
            dataset[exposure] = le.fit_transform(dataset[exposure])
            df_tranf['index'] = [0, 1]
            df_tranf['exposure'] = [str(x) for x in le.classes_]
        if dataset[outcome].dtypes != 'int64':
            le = LabelEncoder()
            dataset[outcome] = le.fit_transform(dataset[outcome])
            df_tranf['index'] = [0, 1]
            df_tranf['outcome'] = [str(x) for x in le.classes_]
        test_status = 'Unable to compute ' + method + ' test for the selected columns. '
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
            return JSONResponse(content={'status': 'Success',
                                     'table': df.to_json(orient="records"), 'col_transormed':df_tranf.to_json(orient="records")},
                                                        status_code=200)
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
            raise Exception
        # print(rr.summary())
        df = rr.results
        df.insert(loc=0, column='Ref:', value=[0,1])
        fig = plt.figure(1)
        ax = plt.subplot(111)
        rr.plot()
        plt.savefig(path_to_storage +"/output/Risktest.svg", format="svg")
        test_status = 'Error in creating info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": method,
                "test_params": {
                    'exposure': exposure,
                    'outcome': outcome,
                    'time': time,
                    'reference': reference,
                    'alpha': alpha
                },
                "test_results": {
                    'table': df.to_dict()
                },
                "Output_datasets": [],
                'Saved_plots': [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                             step_id + '/Risktest.svg'}
                                    ]
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success',
                                     'table': df.to_json(orient="records"), 'col_transormed':df_tranf.to_json(orient="records")},
                                                        status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'table':'[]', 'col_transormed':'[]'},
                            status_code=200)

@router.get("/two_sided_risk_ci")
async def two_sided_risk_ci(workflow_id: str,
                            step_id: str,
                            run_id: str,
                            events: int,
                            total: int,
                            alpha: float | None = Query(default=0.05),
                            confint: str | None = Query("wald",
                                                       regex="^(wald)$|^(hypergeometric)$")):
    test_status = 'Unable to compute the estimated risk.'
    to_return = {'estimated risk': '', 'lower bound': '', 'upper bound': '',
                 'standard error': ''}
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    try:
        r = risk_ci(events=events, total=total, alpha=alpha, confint=confint)
        estimated_risk = r.point_estimate
        lower_bound = r.lower_bound
        upper_bound = r.upper_bound
        standard_error = r.standard_error
        to_return = {'estimated risk': estimated_risk, 'lower bound': lower_bound, 'upper bound': upper_bound, 'standard error': standard_error}
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Linear discriminant analysis',
                "test_params": {'events': str(events),
                                'total': str(total),
                                'alpha': str(alpha),
                                'confint': confint},
                "test_results": to_return,
                "Output_datasets": [],
                "Saved_plots": []
            }
            file_data['results'] |= new_data
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'result': to_return}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'result': to_return}, status_code=200)


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
    test_status = 'Unable to compute the estimated risk.'
    to_return = {'estimated_risk': '', 'lower_bound': '', 'upper_bound': '',
                 'standard_error': ''}
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    try:
        r = risk_ratio(a=exposed_with, b=unexposed_with, c=exposed_without, d=unexposed_without, alpha=alpha)
        estimated_risk = r.point_estimate
        lower_bound = r.lower_bound
        upper_bound = r.upper_bound
        standard_error = r.standard_error
        to_return = {'estimated_risk': estimated_risk, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'standard_error': standard_error}
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Risk_ratio',
                "test_params": {'exposed_with': str(exposed_with),
                                'unexposed_with': str(unexposed_with),
                                'exposed_without': str(exposed_without),
                                'unexposed_without': str(unexposed_without)},
                "test_results": to_return,
                "Output_datasets": [],
                "Saved_plots": []
            }
            file_data['results'] |= new_data
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'result': to_return}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'result': to_return}, status_code=200)

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
    test_status = 'Unable to compute the estimated risk.'
    to_return = {'risk_difference': '', 'lower_bound': '', 'upper_bound': '',
                 'standard_error': ''}
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    try:
        r = risk_difference(a=exposed_with, b=unexposed_with, c=exposed_without, d=unexposed_without, alpha=alpha)
        estimated_risk = r.point_estimate
        lower_bound = r.lower_bound
        upper_bound = r.upper_bound
        standard_error = r.standard_error
        to_return = {'risk_difference': estimated_risk, 'lower_bound': lower_bound, 'upper_bound': upper_bound,
                     'standard_error': standard_error}
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Risk_difference',
                "test_params": {'exposed_with': str(exposed_with),
                                'unexposed_with': str(unexposed_with),
                                'exposed_without': str(exposed_without),
                                'unexposed_without': str(unexposed_without)},
                "test_results": to_return,
                "Output_datasets": [],
                "Saved_plots": []
            }
            file_data['results'] |= new_data
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'result': to_return}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'result': to_return}, status_code=200)


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
    test_status = 'Unable to compute the estimated risk.'
    to_return = {'nnt': '', 'lower_bound': '', 'upper_bound': '',
                 'standard_error': ''}
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    try:
        r = number_needed_to_treat(a=exposed_with, b=unexposed_with, c=exposed_without, d=unexposed_without, alpha=alpha)
        estimated_risk = r.point_estimate
        lower_bound = r.lower_bound
        upper_bound = r.upper_bound
        standard_error = r.standard_error
        to_return = {'nnt': estimated_risk, 'lower_bound': lower_bound, 'upper_bound': upper_bound,
                     'standard_error': standard_error}
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Number_needed_to_treat',
                "test_params": {'exposed_with': str(exposed_with),
                                'unexposed_with': str(unexposed_with),
                                'exposed_without': str(exposed_without),
                                'unexposed_without': str(unexposed_without)},
                "test_results": to_return,
                "Output_datasets": [],
                "Saved_plots": []
            }
            file_data['results'] |= new_data
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'result': to_return}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'result': to_return}, status_code=200)


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
    test_status = 'Unable to compute the estimated risk.'
    to_return = {'odds_ratio': '', 'lower_bound': '', 'upper_bound': '',
                 'standard_error': ''}
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    try:
        r = odds_ratio(a=exposed_with, b=unexposed_with, c=exposed_without, d=unexposed_without, alpha=alpha)
        estimated_risk = r.point_estimate
        lower_bound = r.lower_bound
        upper_bound = r.upper_bound
        standard_error = r.standard_error
        to_return = {'odds_ratio': estimated_risk, 'lower_bound': lower_bound, 'upper_bound': upper_bound,
                     'standard_error': standard_error}
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Odds_ratio',
                "test_params": {'exposed_with': str(exposed_with),
                                'unexposed_with': str(unexposed_with),
                                'exposed_without': str(exposed_without),
                                'unexposed_without': str(unexposed_without)},
                "test_results": to_return,
                "Output_datasets": [],
                "Saved_plots": []
            }
            file_data['results'] |= new_data
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'result': to_return}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'result': to_return}, status_code=200)


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
    test_status = 'Unable to compute the estimated risk.'
    to_return = {'incident_rate_ratio': '', 'lower_bound': '', 'upper_bound': '',
                 'standard_error': ''}
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    try:
        r = incidence_rate_ratio(a=exposed_with, c=unexposed_with, t1=person_time_exposed, t2=person_time_unexposed, alpha=alpha)
        estimated_risk = r.point_estimate
        lower_bound = r.lower_bound
        upper_bound = r.upper_bound
        standard_error = r.standard_error
        to_return = {'incident_rate_ratio': estimated_risk, 'lower_bound': lower_bound, 'upper_bound': upper_bound,
                     'standard_error': standard_error}
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Incidence_rate_ratio',
                "test_params": {'exposed_with': str(exposed_with),
                                'unexposed_with': str(unexposed_with),
                                'person_time_exposed': str(person_time_exposed),
                                'person_time_unexposed': str(person_time_unexposed)},
                "test_results": to_return,
                "Output_datasets": [],
                "Saved_plots": []
            }
            file_data['results'] |= new_data
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'result': to_return}, status_code=200)

    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'result': to_return}, status_code=200)


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
    test_status = 'Unable to compute the estimated risk.'
    to_return = {'incident_rate_difference': '', 'lower_bound': '', 'upper_bound': '',
                 'standard_error': ''}
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    try:
        r = incidence_rate_difference(a=exposed_with, c=unexposed_with, t1=person_time_exposed, t2=person_time_unexposed, alpha=alpha)
        estimated_risk = r.point_estimate
        lower_bound = r.lower_bound
        upper_bound = r.upper_bound
        standard_error = r.standard_error
        to_return = {'incident_rate_difference': estimated_risk, 'lower_bound': lower_bound, 'upper_bound': upper_bound,
                     'standard_error': standard_error}
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Incident_rate_difference',
                "test_params": {'exposed_with': str(exposed_with),
                                'unexposed_with': str(unexposed_with),
                                'person_time_exposed': str(person_time_exposed),
                                'person_time_unexposed': str(person_time_unexposed)},
                "test_results": to_return,
                "Output_datasets": [],
                "Saved_plots": []
            }
            file_data['results'] |= new_data
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'result': to_return}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'result': to_return}, status_code=200)


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
    dfv = pd.DataFrame()
    # dfe = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = column_2
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        test_status='Unable to retrieve datasets'
        data = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        column_2 = dfv['Variable'].tolist()
        selected_columns = pd.unique(dfv['Variable'])
        for column in data.columns:
            if column not in selected_columns:
                data = data.drop(str(column), axis=1)

        test_status = 'Unable to compute ' + method+' correlation.'
        df = data[column_2]
        # Not for all methods -
        # df1 = df.rcorr(stars=False).round(5)
        # corrs = df.corr()

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
                print(res)
                count = count + 1
                for ind, row in res.iterrows():
                    temp_to_append = {
                        "id": count,
                        "Cor": row['Cor'],
                        "n": row['n'],
                        "r": row['r'] if not pd.isna(row['r']) else 'NaN',
                        "CI95%": "[" + str(row['CI95%'].item(0)) + "," + str(row['CI95%'].item(1)) + "]" if type(row['CI95%'])!=float else 'NaN',
                        "p-val": 'NaN' if pd.isna(row['p-val']) else row['p-val'],
                        "power": 'NaN' if pd.isna(row['power']) else row['power']
                    }
                    if method == 'pearson':
                        temp_to_append["BF10"] = row['BF10']
                    if method == 'shepherd':
                        temp_to_append["outliers"] = row['outliers']
                all_res.append(temp_to_append)
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": "Correlation test",
                    "test_params": {
                        'selected_method': method,
                        'selected_variable': column_2,
                        'alternative': alternative
                    },
                    "test_results": all_res
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status':'Success', 'DataFrame': all_res}, status_code=200)
        # return JSONResponse(content={'status':'Success', 'DataFrame': all_res, "Table_rcorr": df1.to_json(orient='records')}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status':test_status, 'DataFrame': []}, status_code=200)
        # return JSONResponse(content={'status':test_status, 'DataFrame': [],'Table_rcorr':dfe.to_json(orient='records')}, status_code=200)

# TODO:We use statsModel
# @router.get("/linear_regressor_pinguin")
# async def linear_regression_pinguin(dependent_variable: str,
#                                     alpha: float | None=Query(default=0.05),
#                                     relimp: bool | None=Query(default=False),
#                                     independent_variables: list[str] | None = Query(default=None)):
#
#     lm = pingouin.linear_regression(data[independent_variables], data[dependent_variable], as_dataframe=True, alpha=alpha, relimp=relimp)
#
#     return {'residuals': lm.residuals_.tolist(), 'degrees of freedom of the model': lm.df_model_, 'degrees of freedom of the residuals': lm.df_resid_ , 'dataframe': lm.to_json(orient='split')}

@router.get("/logistic_regressor_pinguin")
async def logistic_regression_pinguin(workflow_id: str, step_id: str, run_id: str,
                                      dependent_variable: str,
                                      alpha: float | None= Query(default=0.05),
                                      independent_variables: list[str] | None = Query(default=None)):
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Please provide all mandatory fields (dataset, dependent variable, one or more independent variables)'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        independent_variables = list(dfv['Variable'].values)
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        df_label = dataset[dependent_variable]


        try:
            dataset[dependent_variable] = dataset[dependent_variable].astype('int')
        except Exception as e:
            binary_values = dataset[dependent_variable].unique()

            dataset[dependent_variable].replace(binary_values[0], 0, inplace=True)
            dataset[dependent_variable].replace(binary_values[1], 1, inplace=True)


        # for columns in dataset.columns:
        #     if columns not in independent_variables:
        #         dataset = dataset.drop(str(columns), axis=1)
        # data.dropna(inplace=True)

        lm = pingouin.logistic_regression(dataset[independent_variables], dataset[dependent_variable], as_dataframe=True,
                                          alpha=alpha)

        lm.rename(columns={'CI[2.5%]': 'CI_low', 'CI[97.5%]': 'CI_high'}, inplace=True)

        values_dict = {}
        for column in lm.columns:
            values_dict[column] = list(lm[column])

        response = {'dataframe': lm.to_json(orient='records'), 'values_dict': values_dict, 'values_columns': list(lm.columns)}

        lm.to_csv(path_to_storage + '/output/logistic_pingouin.csv', index=False)
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Logistic Regression Pingouin',
                "test_params": {
                    'dependent variable': dependent_variable,
                    'independent variables': independent_variables,
                    'alpha': alpha,
                },
                "test_results": {'dataframe': lm.to_json(orient='records')}
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/logistic_pingouin.csv'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Result': response},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'Result': '[]'},
                            status_code=200)

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
    df = pd.DataFrame()
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets

    try:
        test_status = 'Please provide all mandatory fields (dataset, dependent variable, one or more independent variables)'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        independent_variables = dfv['Variable'].tolist()
        data = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        # data.dropna(inplace=True)

        x = data[independent_variables]
        y = data[dependent_variable]

        df_dict = {}
        for name in independent_variables:
            df_dict[str(name)] = data[str(name)]

        df_dict[str(dependent_variable)] = data[dependent_variable]
        df_features_label = pd.DataFrame.from_dict(df_dict)
        test_status = 'Unable to perform linear regression'
        x = sm.add_constant(x)

        # if regularization:
        #     model = sm.OLS(y, x).fit_regularized(method='elastic_net')
        # else:
        # fig = plt.figure(1)
        model = sm.OLS(y, x).fit()

        fitted_value = model.fittedvalues
        df_fitted_value = pd.DataFrame(fitted_value, columns=['fitted_values'])
        resid_value = model.resid
        df_resid_value = pd.DataFrame(resid_value, columns=['residuals'])
        # create instance of influence
        influence = model.get_influence()
        # sm.graphics.influence_plot(model)
        # plt.show()

        # obtain standardized residuals
        standardized_residuals = influence.resid_studentized_internal
        inf_sum = influence.summary_frame()
        df_final_influence = pd.concat([df_features_label, inf_sum, df_fitted_value, df_resid_value], axis=1)
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
        df_0.rename(columns={1: 'Values'}, inplace=True)
        df_0 = df_0.iloc[:-2]
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

    # if not regularization:
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

        z = het_goldfeldquandt(y, x)
        labels = ['F-statistic', 'p-value', "ordering used in the alternative"]
        results_goldfeldquandt = dict(zip(labels, z))
        goldfeld_test = pd.DataFrame(results_goldfeldquandt.values(), index=results_goldfeldquandt.keys())
        goldfeld_test.rename(columns={0: 'Values'}, inplace=True)

        # print(inf_dict)
        df_final_influence.to_csv(path_to_storage + '/output/influence_points.csv', index=False)

        response = {'DataFrame with all available influence results': df_final_influence.to_html(),
                    'first_table': df_0.to_json(orient='split'), 'second table': df_1.to_html(),
                    'third table': df_2.to_dict(), 'dataframe white test': white_test.to_json(orient='split'),
                    'dep': df_0.loc['Dep. Variable:'][0], 'model': df_0.loc['Model:'][0],
                    'method': df_0.loc['Method:'][0], 'date': df_0.loc['Date:'][0],
                    'time': df_0.loc['Time:'][0], 'no_obs': df_0.loc['No. Observations:'][0],
                    'resid': df_0.loc['Df Residuals:'][0],
                    'df_model': df_0.loc['Df Model:'][0], 'cov_type': df_0.loc['Covariance Type:'][0],
                    'r_squared': df_0.loc['R-squared:'][0], 'adj_r_squared': df_0.loc['Adj. R-squared:'][0],
                    'f_stat': df_0.loc['F-statistic:'][0], 'prob_f': df_0.loc['Prob (F-statistic):'][0],
                    'log_like': df_0.loc['Log-Likelihood:'][0], 'aic': df_0.loc['AIC:'][0],
                    'bic': df_0.loc['BIC:'][0],
                    'omnibus': df_2.loc['Omnibus:'][0], 'prob_omni': df_2.loc['Prob(Omnibus):'][0],
                    'skew': df_2.loc['Skew:'][0], 'kurtosis': df_2.loc['Kurtosis:'][0],
                    'durbin': df_2.loc['Durbin-Watson:'][0], 'jb': df_2.loc['Jarque-Bera (JB):'][0],
                    'prob_jb': df_2.loc['Prob(JB):'][0], 'cond': df_2.loc['Cond. No.'][0],
                    'test_stat': white_test.loc['Test Statistic'][0],
                    'test_stat_p': white_test.loc['Test Statistic p-value'][0],
                    'white_f_stat': white_test.loc['F-Statistic'][0],
                    'white_prob_f': white_test.loc['F-Test p-value'][0],
                    'influence_columns': list(df_final_influence.columns), 'influence_dict': inf_dict,
                    'bresuch_lagrange': bresuch_test.loc['Lagrange multiplier statistic'][0],
                    'bresuch_p_value': bresuch_test.loc['p-value'][0],
                    'bresuch_f_value': bresuch_test.loc['f-value'][0],
                    'bresuch_f_p_value': bresuch_test.loc['f p-value'][0],
                    'Goldfeld-Quandt F-value': goldfeld_test.loc['F-statistic'][0],
                    'Goldfeld-Quandt p-value': goldfeld_test.loc['p-value'][0],
                    'Goldfeld-Quandt ordering used in the alternative':
                        goldfeld_test.loc['ordering used in the alternative'][0]}

        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Linear Regression',
                "test_params": {
                    'dependent variable': dependent_variable,
                    'independent variables': independent_variables
                },
                "test_results": {'coefs': df_1.to_dict(),
                                 'white test': white_test.to_dict(),
                                 'dep': df_0.loc['Dep. Variable:'][0], 'model': df_0.loc['Model:'][0],
                                 'method': df_0.loc['Method:'][0], 'date': df_0.loc['Date:'][0],
                                 'time': df_0.loc['Time:'][0], 'no_obs': df_0.loc['No. Observations:'][0],
                                 'resid': df_0.loc['Df Residuals:'][0],
                                 'df_model': df_0.loc['Df Model:'][0], 'cov_type': df_0.loc['Covariance Type:'][0],
                                 'r_squared': df_0.loc['R-squared:'][0], 'adj_r_squared': df_0.loc['Adj. R-squared:'][0],
                                 'f_stat': df_0.loc['F-statistic:'][0], 'prob_f': df_0.loc['Prob (F-statistic):'][0],
                                 'log_like': df_0.loc['Log-Likelihood:'][0], 'aic': df_0.loc['AIC:'][0],
                                 'bic': df_0.loc['BIC:'][0],
                                 'omnibus': df_2.loc['Omnibus:'][0], 'prob_omni': df_2.loc['Prob(Omnibus):'][0],
                                 'skew': df_2.loc['Skew:'][0], 'kurtosis': df_2.loc['Kurtosis:'][0],
                                 'durbin': df_2.loc['Durbin-Watson:'][0], 'jb': df_2.loc['Jarque-Bera (JB):'][0],
                                 'prob_jb': df_2.loc['Prob(JB):'][0], 'cond': df_2.loc['Cond. No.'][0],
                                 'test_stat': white_test.loc['Test Statistic'][0],
                                 'test_stat_p': white_test.loc['Test Statistic p-value'][0],
                                 'white_f_stat': white_test.loc['F-Statistic'][0],
                                 'white_prob_f': white_test.loc['F-Test p-value'][0],
                                 'bresuch_lagrange': bresuch_test.loc['Lagrange multiplier statistic'][0],
                                 'bresuch_p_value': bresuch_test.loc['p-value'][0],
                                 'bresuch_f_value': bresuch_test.loc['f-value'][0],
                                 'bresuch_f_p_value': bresuch_test.loc['f p-value'][0],
                                 'Goldfeld-Quandt F-value': goldfeld_test.loc['F-statistic'][0],
                                 'Goldfeld-Quandt p-value': goldfeld_test.loc['p-value'][0],
                                 'Goldfeld-Quandt ordering used in the alternative':
                                    goldfeld_test.loc['ordering used in the alternative'][0]}
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/influence_points.csv'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status':'Success', 'Result': response},
                            status_code=200)
    except Exception as e:
        return JSONResponse(content={'status': test_status, 'Result': '[]'},
                            status_code=200)


@router.get("/z_score")
async def z_score(workflow_id: str,
                  step_id: str,
                  run_id: str,
                  ddof: int | None = Query(default=0),
                  nan_policy: Optional[str] | None = Query("propagate",
                                                           regex="^(propagate)$|^(raise)$|^(omit)$"),
                  dependent_variables: list[str] | None = Query(default=None)):
    df = pd.DataFrame()
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = dependent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        print(selected_datasources)
        test_status = 'Unable to retrieve datasets'
        selected_columns = pd.unique(dfv['Variable'])
        print(selected_columns)
        for ds in selected_datasources:
            dataset = load_data_from_csv(path_to_storage + "/" + ds)
            # Keep requested Columns
            for columns in dataset.columns:
                if columns not in selected_columns:
                    dataset = dataset.drop(str(columns), axis=1)
            # Get min values
            test_status = 'Unable to compute the z score values for the selected columns'
            for column in dataset.columns:
                try:
                    df[column] = zscore(dataset[column], ddof=ddof, nan_policy=nan_policy)
                except:
                    df[column] = np.nan

        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Z score',
                "test_params": dependent_variables,
                "test_results": df.to_dict()
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Dataframe': df.to_json(orient="records")},
                            status_code=200)

    except Exception as e:
        df["Error"] = test_status
        print(e)
        return JSONResponse(content={'status': test_status, 'Dataframe': df.to_json(orient="records")},
                            status_code=200)
    # x = data[dependent_variable]
    #
    # z_score_res = zscore(x)

    return {'z_score': list(z_score_res)}

@router.get("/logistic_regressor_statsmodels")
async def logistic_regression_statsmodels(workflow_id: str, step_id: str, run_id: str,
                                          dependent_variable: str,
                                          independent_variables: list[str] | None = Query(default=None)):
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Please provide all mandatory fields (dataset, dependent variable, one or more independent variables)'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        independent_variables = list(dfv['Variable'].values)
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        df_label = dataset[dependent_variable]

        try:
            dataset[dependent_variable] = dataset[dependent_variable].astype('int')
        except Exception as e:
            binary_values = dataset[dependent_variable].unique()

            dataset[dependent_variable].replace(binary_values[0], 0, inplace=True)
            dataset[dependent_variable].replace(binary_values[1], 1, inplace=True)

        # for columns in dataset.columns:
        #     if columns not in independent_variables:
        #         dataset = dataset.drop(str(columns), axis=1)
        # data.dropna(inplace=True)

        x = dataset[independent_variables]
        y = dataset[dependent_variable]

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
        new_header = new_header.tolist()
        print(new_header)
        new_header.insert(0, 'names')
        df_1 = df_1[1:]
        # df_1.set_index(0, inplace=True)
        df_1.columns = new_header
        df_1.rename(columns={'[0.025': 'CI_low', '0.975]': 'CI_high'}, inplace=True)
        # df_1.index.name = None

        df_1.fillna('', inplace=True)
        values_dict = {}
        for column in df_1.columns:
            values_dict[column] = list(df_1[column])

        print(df_0.loc['Pseudo R-squ.:'][0])
        print(df_0.loc['LLR p-value:'][0])


        response = {'first_table': df_0.to_json(orient='records'), 'second table': df_1.to_json(orient='records'),
            'values_dict': values_dict, 'values_columns': list(df_1.columns),
            'dep': df_0.loc['Dep. Variable:'][0], 'model': df_0.loc['Model:'][0],
            'method': df_0.loc['Method:'][0], 'date': df_0.loc['Date:'][0],
            'time': df_0.loc['Time:'][0], 'no_obs': df_0.loc['No. Observations:'][0],
            'resid': df_0.loc['Df Residuals:'][0],
            'df_model': df_0.loc['Df Model:'][0], 'cov_type': df_0.loc['Covariance Type:'][0],
            'pseudo_r_squared': df_0.loc['Pseudo R-squ.:'][0], 'log_like': df_0.loc['Log-Likelihood:'][0],
            'LL-Null': df_0.loc['LL-Null:'][0], 'LLR p-value': df_0.loc['LLR p-value:'][0],
            'converged': df_0.loc['converged:'][0]}

        df_1.to_csv(path_to_storage + '/output/logistic_statsmodels.csv', index=False)
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Logistic Regression Statsmodels',
                "test_params": {
                    'dependent variable': dependent_variable,
                    'independent variables': independent_variables,
                },
                "test_results": {'dep': df_0.loc['Dep. Variable:'][0], 'model': df_0.loc['Model:'][0],
            'method': df_0.loc['Method:'][0], 'date': df_0.loc['Date:'][0],
            'time': df_0.loc['Time:'][0], 'no_obs': df_0.loc['No. Observations:'][0],
            'resid': df_0.loc['Df Residuals:'][0],
            'df_model': df_0.loc['Df Model:'][0], 'cov_type': df_0.loc['Covariance Type:'][0],
            'pseudo_r_squared': df_0.loc['Pseudo R-squ.:'][0], 'log_like': df_0.loc['Log-Likelihood:'][0],
            'LL-Null': df_0.loc['LL-Null:'][0], 'LLR p-value': df_0.loc['LLR p-value:'][0],
            'converged': df_0.loc['converged:'][0]}
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/logistic_statsmodels.csv'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Result': response},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'Result': '[]'},
                            status_code=200)

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

    # dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    df = pd.DataFrame()
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        # Keep requested Columns
        selected_columns = pd.unique(dfv['Variable'])
        for columns in dataset.columns:
            if columns not in selected_columns:
                dataset = dataset.drop(str(columns), axis=1)
        test_status = 'Unable to compute the Covariance matrix for the selected columns'
        res = statisticsCov(dataset, ddof)
        if len(res) >= 1:
            df = pd.DataFrame(res, columns=dataset.columns)
        else:
            df = ["N/A"]
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": 'Covariance matrix',
                    "test_params": independent_variables,
                    "test_results":df.to_dict()
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'Dataframe': df.to_json(orient="records")},
                                status_code=200)
    except Exception as e:
        df["Error"] = test_status
        print(e)
        return JSONResponse(content={'Dataframe': df.to_json(orient="records")},
                                status_code=200)

    # r = cov(dataset, ddof=ddof)
    #
    # return {'Covariance Matrix': r.tolist()}

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
    print(type(original_eigen_values))



    fig = plt.figure(1)

    plt.scatter(range(1, dataset.shape[1] + 1), original_eigen_values)
    plt.plot(range(1, dataset.shape[1] + 1), original_eigen_values)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + '/output/factor_eigen_values.png')
    plt.show()

    # html_str = mpld3.fig_to_html(fig)
    df_orig_eigen_values = pd.DataFrame({'Factors': ['Factor' + str(i) for i in range(len(independent_variables))],
                                          'Original Eigen Values': original_eigen_values})
    to_return = {'df_orig_eigen_values': df_orig_eigen_values.to_json(orient='records')}

    return to_return


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

    # dataset = load_file_csv_direct(workflow_id, run_id, step_id)
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    print(workflow_id, step_id, run_id, independent_variables)
    test_status = ''
    dfi = pd.DataFrame()
    try:
        test_status = 'Dataset is not defined'
        dfi['variables'] = independent_variables
        dfi[['Datasource', 'Variable']] = dfi["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        independent_variables.clear()
        independent_variables = dfi['Variable'].tolist()
        test_status = 'Unable to retrieve datasets'
        selected_datasources = pd.unique(dfi['Datasource'])
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        print(selected_datasources)
        print(independent_variables)
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

        # return to_return
        return JSONResponse(content={'status': 'Success', 'Result': to_return},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'Result': '[]'},
                            status_code=200)


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
                             mediator: list[str] | None = Query(default=None),
                             independent: list[str] | None = Query(default=None)):
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    dfm = pd.DataFrame()
    dfi = pd.DataFrame()
    print(workflow_id, step_id, run_id, dependent_1, exposure, mediator, independent)
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        selected_datasource = dependent_1.split("--")[0]
        dependent_1 = dependent_1.split("--")[1]
        exposure = exposure.split("--")[1]
        dfm['variables'] = mediator
        dfm[['Datasource', 'Variable']] = dfm["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        mediator = dfm['Variable'].tolist()
        if independent != ['']:
            dfi['variables'] = independent
            dfi[['Datasource', 'Variable']] = dfi["variables"].apply(lambda x: pd.Series(str(x).split("--")))
            independent = dfi['Variable'].tolist()
        if (dependent_1 == exposure) or (dependent_1 in mediator) or (exposure in mediator):
            test_status = 'Select different columns for outcome, predictor and mediator variables'
            raise Exception
        elif (dependent_1 in independent) or (exposure in independent):
            test_status = 'Select different columns for outcome, predictor and covar variables'
            raise Exception
        for med in mediator:
            if med in independent:
                test_status = 'Mediator columns cannot be in covar variables'
                raise Exception
        test_status = 'Unable to retrieve datasets'
        data = load_data_from_csv(path_to_storage + "/" + selected_datasource)

        # We want X to affect Y. If there is no relationship between X and Y, there is nothing to mediate.
        # model.0 <- lm(Y ~ X, myData)


        # We use penguin
        # if mediator != ['']:
        #     print("no med")
        #     dfm['variables'] = mediator
        #     dfm[['Datasource', 'Variable']] = dfm["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        #     mediator = dfm['Variable'].tolist()
        #     z_m = ''
        #     for i in range(len(mediator)):
        #         z_m = z_m + "+" + mediator[i] if z_m != '' else mediator[i]
        # if independent != ['']:
        #     print("no ind")
        #     dfi['variables'] = independent
        #     dfi[['Datasource', 'Variable']] = dfi["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        #     independent = dfi['Variable'].tolist()
        #     z = ''
        #     for i in range(len(independent)):
        #         z = z + "+" + independent[i] if z != '' else independent[i]
        # output_str = dependent_1 + "~" + exposure + "+" + z if z!='' else dependent_1 + "~" + exposure
        #         print("output_str = " + output_str)
        #         model0 = sm.GLM.from_formula(output_str, data)
        # m1 = model0.fit()
        # df = m1.summary()
        # We want X to affect M. If X and M have no relationship, M is just a third variable that may or may not
        # be associated with Y. A mediation makes sense only if X affects M.
        # model.M <- lm(M ~ X, myData)

        # mediator_str = z_m + "~" + exposure
        # print("mediator_str = "+mediator_str)
        # mediator_model = sm.OLS.from_formula(mediator[i]+ "~" + exposure, data)
        # res = mediator_model.fit()

        # We want M to affect Y, but X to no longer affect Y (or X to still affect Y but in a smaller
        # magnitude). If a mediation effect exists, the effect of X on Y will disappear
        # (or at least weaken) when M is included in the regression. The effect of X on Y goes through M.
        # model.Y <- lm(Y ~ X + M, myData)
        # outcome_str = dependent_1 + "~" + exposure + "+" + z_m + "+" + z
        # print("outcome_str = "+outcome_str)
        # outcome_model = sm.GLM.from_formula(outcome_str, data)
        # res = outcome_model.fit()
        # print(res.summary())
        # Call analysis with the models
        # results <- mediate(model.M, model.Y, treat='X', mediator='M',
        #                    boot=TRUE, sims=500)
        # it accepts only one Mediator
        # med = Mediation(outcome_model, mediator_model, exposure, mediator).fit()
        # df = med.summary()

        test_status = 'Unable to compute Mediation Analysis.'
        # df1, dist = mediation_analysis(data=data, x=exposure, m=mediator, y=dependent_1,
        #                          covar=independent, seed=42,return_dist=True)
            # .round(3)
        # df1.columns = df1.columns.str.replace('.', ',', regex=True)
        if independent != ['']:
            df, dist = mediation_analysis(data=data, x=exposure, m=mediator, y=dependent_1,
                                           covar=independent, seed=42, return_dist=True)
        else:
            df, dist = mediation_analysis(data=data, x=exposure, m=mediator, y=dependent_1,
                                           seed=42, return_dist=True)
        df.columns = df.columns.str.replace('.', ',', regex=True)
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax1 = fig.add_subplot()
        # ax = sns.kdeplot(dist[0])
        # ax1 =sns.kdeplot(dist[1])
        # plt.show()
        # sns.kdeplot(df["flipper_length_mm"])
        # print(dist)
        # print(df)
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": "Mediation analysis",
                    "test_params": {
                        'outcome variable': dependent_1,
                        'predictor variable': exposure,
                        'mediator variable': mediator,
                        'independent variable': independent
                    },
                    "test_results": df.to_dict()
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status':'Success', 'Result': df.to_json(orient='records')},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'Result': '[]'},
                            status_code=200)

@router.get("/canonical_correlation_analysis")
async def canonical_correlation(workflow_id: str,
                                step_id: str,
                                run_id: str,
                                n_components: int | None = Query(default=2),
                                independent_variables_1: list[str] | None = Query(default=None),
                                independent_variables_2: list[str] | None = Query(default=None)):
    dfv = pd.DataFrame()
    dfv2 = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables1'] = independent_variables_1
        dfv[['Datasource', 'Variable1']] = dfv["variables1"].apply(lambda x: pd.Series(str(x).split("--")))
        dfv2['variables2'] = independent_variables_2
        dfv2[['Datasource', 'Variable2']] = dfv2["variables2"].apply(lambda x: pd.Series(str(x).split("--")))
        independent_variables_1 = dfv["Variable1"].tolist()
        independent_variables_2 = dfv2["Variable2"].tolist()
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Variables cannot be found in the same Dataset'
        if selected_datasources != pd.unique(dfv2['Datasource']):
            print(selected_datasources+"      vs     "+pd.unique(dfv2['Datasource']))
            raise Exception

        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])

        X = dataset[dataset.columns.intersection(independent_variables_1)]
        Y = dataset[dataset.columns.intersection(independent_variables_2)]

        test_status = 'Unable to compute Canonical correlation for the selected columns.'
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
        coef_df = pd.DataFrame(np.round(my_cca.coef_, 5), columns=Y.columns)
        coef_df.index = X.columns
        plt.figure(figsize=(5, 5))
        s= sns.heatmap(coef_df, cmap='coolwarm', annot=True, linewidths=1, vmin=-1)
        s.set(xlabel='Y samle', ylabel='X sample')
        # plt.title = "CCA coefficients."
        plt.savefig(path_to_storage + "/output/CCA_coefs.svg", format="svg")
        plt.show()

        xweights = pd.DataFrame(my_cca.x_weights_, columns=comp_titles)
        xweights.insert(loc=0, column='Feature', value=independent_variables_1)
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
        Xc_df.to_csv(path_to_storage + '/output/Xc_df.csv', index=False)
        Yc_df.to_csv(path_to_storage + '/output/Yc_df.csv', index=False)

        test_status = 'Erro in creating info file.'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": 'Canonical correlation',
                    "test_params": {
                        "Training vectors": independent_variables_1,
                        "Target vectors": independent_variables_2
                    },
                    "test_results":{'xweights': xweights.to_dict(),
                                     'yweights': yweights.to_dict(),
                                     'xloadings': xloadings.to_dict(),
                                     'yloadings': yloadings.to_dict(),
                                     'xrotations': xrotations.to_dict(),
                                     'yrotations': yrotations.to_dict(),
                                     'coef_df': coef_df.to_dict()}
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                step_id + '/Xc_df.csv'},
                                            {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                     step_id + '/Yc_df.csv'}
                                            ]
            file_data['Saved_plots'] = [{"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                             step_id + '/CCA_XYcorr.svg'},
                                        {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                 step_id + '/CCA_comp_corr.svg'},
                                        {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                 step_id + '/CCA_XY_c_corr.svg'},
                                        {"file": 'workflows/' + workflow_id + '/' + run_id + '/' +
                                                 step_id + '/CCA_coefs.svg'}]
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success',
                                     'xweights': xweights.to_json(orient='records'),
                                     'yweights': yweights.to_json(orient='records'),
                                     'xloadings': xloadings.to_json(orient='records'),
                                     'yloadings': yloadings.to_json(orient='records'),
                                     'xrotations': xrotations.to_json(orient='records'),
                                     'yrotations': yrotations.to_json(orient='records'),
                                     'coef_df': coef_df.to_json(orient='records'),
                                     'Xc_df': Xc_df.to_json(orient='records'),
                                     'Yc_df': Yc_df.to_json(orient='records')},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status,
                                     'xweights': "[]",
                                     'yweights': "[]",
                                     'xloadings': "[]",
                                     'yloadings': "[]",
                                     'xrotations': "[]",
                                     'yrotations': "[]",
                                     'coef_df': "[]",
                                     'Xc_df': "[]",
                                     'Yc_df': "[]"},
                            status_code=200)

@router.get("/granger_analysis")
async def compute_granger_analysis(workflow_id: str,
                                   step_id: str,
                                   run_id: str,
                                   predictor_variable: str,
                                   response_variable: str,
                                   num_lags: list[int] | None = Query(default=None)):
                                   # all_lags_up_to : bool | None = Query(default=False)):

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        selected_datasource, predictor_variable = predictor_variable.split("--")
        response_variable = response_variable.split("--")[1]
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasource)

        # if all_lags_up_to==False:
        #     print(grangercausalitytests(dataset[[response_variable, predictor_variable]], maxlag=[num_lags]))
        # else:
        test_status = 'Unable to conduct granger causality test. Check that all parameters have been provided'
        if len(num_lags) == 1:
            granger_result = grangercausalitytests(dataset[[response_variable, predictor_variable]], maxlag=num_lags[0])
        else:
            granger_result = grangercausalitytests(dataset[[response_variable, predictor_variable]], maxlag=num_lags)
        # Union[int, List[int]]

        # print(type(granger_result))
        # # df_granger = pd.DataFrame(data=granger_result)
        # print(granger_result.keys())
        lag_numbers = list(granger_result.keys())
        to_return = []
        for lag in lag_numbers:
            func = lambda x : int(x) if type(x) == np.int32 else x
            start_dict = dict((key, list(map(func, tuple_))) for (key, tuple_) in granger_result[lag][0].items())
            list_for_lag = []
            for key, value in start_dict.items():
               if len(value) == 3:
                   list_for_lag.append({"key": key, "F" : "-", "chi2" : "%.3f"%float(round(value[0], 3)), "p": "%.3f"%float(round(value[1], 3)), "df" :str(int(value[2])), "df_denom" : "-", "df_num" : "-"})
               else:
                   list_for_lag.append({"key": key, "F" : "%.3f"%float(round(value[0], 3)), "chi2" : "-", "p": "%.3f"%float(round(value[1], 3)), "df" : "-", "df_denom" : str(int(value[2])), "df_num" : str(int(value[3]))})
            to_return.append({"lag_num" : str(lag), "result": list_for_lag})


        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Granger Analysis',
                "test_params": {
                    'selected_predictor_variable': predictor_variable,
                    'selected_response_variable': response_variable,
                    'selected_num_lags': num_lags,
                },
                "test_results": to_return,
                "Output_datasets": [],
                'Saved_plots': []
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': "Success", 'lags': to_return},
                            status_code=200)
    except Exception as e:
        return JSONResponse(content={'status': test_status, 'lags': []},
                            status_code=200)
    # return to_return



@router.get("/calculate_one_way_welch_anova")
async def compute_one_way_welch_anova(workflow_id: str,
                                      step_id: str,
                                      run_id: str,
                                      dv: str,
                                      between: str):

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        selected_datasource, dv = dv.split("--")
        between = between.split("--")[1]
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        df_data = load_data_from_csv(path_to_storage + "/" + selected_datasource)

        test_status = 'Unable to compute Welch Anova test for the selected columns.'
        df = pingouin.welch_anova(data=df_data, dv=dv, between=between)
        print(df)
        df = df.fillna('')
        all_res = []
        for ind, row in df.iterrows():
            temp_to_append = {
                'id': ind,
                'Source': row['Source'],
                'ddof1': row['ddof1'],
                'ddof2': row['ddof2'],
                'F': row['F'],
                'p-unc': row['p-unc'],
                'np2': row['np2']
            }
            all_res.append(temp_to_append)
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Welch Anova test',
                "test_params": {
                    'selected_depedent_variable': dv,
                    'selected_between_factor':between,
                },
                "test_results": all_res,
                "Output_datasets":[],
                'Saved_plots': []
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success','DataFrame': all_res},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'Dataframe': []},
                            status_code=200)
@router.get("/calculate_kruskal_pinguin")
async def compute_kruskal(workflow_id: str,
                          step_id: str,
                          run_id: str,
                          dependent_variable: str,
                          between_factor: str):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    df = pingouin.kruskal(data=dataset, dv=dependent_variable, between=between_factor)
    print(df)

@router.get("/calculate_anova_repeated_measures_pingouin")
async def compute_anova_repeated_measures_pinguin(workflow_id: str,
                                                  step_id: str,
                                                  run_id: str,
                                                  dv: str,
                                                  subject: str,
                                                  correction: str | None = Query(default=True),
                                                  within: list[str] | None = Query(default=None),
                                                  effsize: str | None = Query("np2",
                                                                             regex="^(np2)$|^(n2)$|^(ng2)$")
                                                  ):

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        selected_datasource = dv.split("--")[0]
        dependent_variable = dv.split("--")[1]
        subject = subject.split("--")[1]

        within = list(map(lambda x: str(x).split("--")[1], within))

        correction = (correction == 'True')

        test_status = 'A column can not be selected multiple times'
        var_list = [dependent_variable] + within + [subject]
        assert len(var_list) == len(set(var_list))

        test_status = 'Unable to retrieve datasets'
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasource)
        pd.set_option('display.max_columns', None)

        df = pingouin.rm_anova(data=dataset, dv=dependent_variable, subject=subject, within=within, correction=correction, effsize=effsize)
        df = df.fillna('')
        print(df)

        columns = [{
            "col": "id"}]

        for col in df.columns:
            match col:
                case "ddof1":
                    new_col = "numerator - DoF"
                case "ddof2":
                    new_col = "denominator - DoF"
                case "ng2" | "n2" | "np2":
                    new_col = "Effect size"
                case "p-unc":
                    new_col = "p-uncorrected"
                case "p-GG-corr":
                    new_col = "Gr.-Geis. corrected p-value"
                case "SS":
                    new_col = "Sums of squares"
                case "DF":
                    new_col = "Degrees of freedom"
                case "MS":
                    new_col = "Mean squares"
                case "eps":
                    new_col = "Epsilon Factor"
                case "W-spher":
                    new_col = "Sphericity stat."
                case "p-spher":
                    new_col = "Sphericity p-value"
                case _:
                    new_col = col
            columns.append({
                "col": new_col
            })
            df.rename(columns={col: new_col}, inplace=True)

        print(columns)

        all_res = []
        for ind, row in df.iterrows():
            temp_to_append = row.to_dict()
            temp_to_append['id'] = ind
            all_res.append(temp_to_append)

        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Anova Repeated Measures',
                "test_params": {
                    'selected_depedent_variable': dependent_variable,
                    'selected_subject_variable':subject,
                    'selected_within_variables':within,
                    'selected_correction': correction,
                    'selected_effsize': effsize,
                },
                "test_results": all_res,
                "Output_datasets":[],
                'Saved_plots': []
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()

        return JSONResponse(content={'status': 'Success', 'DataFrame': all_res, "Columns": columns},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status, 'DataFrame': [], "Columns": []},
                            status_code=200)


@router.get("/calculate_friedman_test_pinguin")
async def compute_friedman_test_pinguin(workflow_id: str,
                                        step_id: str,
                                        run_id: str,
                                        dependent_variable: str,
                                        subject: str,
                                        within: str,
                                        method: str | None = Query("chisq",
                                                                    regex="^(chisq)$|^(f)$")):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    df = pingouin.friedman(data=dataset, dv=dependent_variable, subject=subject, within=within, method=method)
    print(df)

@router.get("/calculate_mixed_anova")
async def compute_mixed_anova_pinguin(workflow_id: str,
                                        step_id: str,
                                        run_id: str,
                                        dependent_variable: str,
                                        subject: str,
                                        within: str,
                                        between: str,
                                        correction: str | None = Query("True",
                                                                       regex="^(True)$|^(auto)$"),
                                        effsize: str | None = Query("np2",
                                                                    regex="^(np2)$|^(n2)$|^(ng2)$")):
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        selected_datasource = dependent_variable.split("--")[0]
        dependent_variable = dependent_variable.split("--")[1]
        subject = subject.split("--")[1]
        within = within.split("--")[1]
        between = between.split("--")[1]
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasource)
        test_status = 'Unable to compute mixed Anova test for the selected columns.'

        # dataset = pingouin.read_dataset('mixed_anova')
        # print(dataset)
        # print(dataset.dtypes)
        # TODO in SPSS they check repeated measures before this
        # TODO check between factor for nans
        # TODO check if within and between factors are categorical
        # check_for_nan = dataset['Group'].isnull().values.any()
        # print(check_for_nan)
        # if correction_1==True:
        print("DATASET", dataset)
        print("DEP_VAR", dependent_variable)
        print("SUBJECT", subject)
        print("WITHIN", within)
        print("BETWEEN", between)

        dataset = dataset[[dependent_variable, subject, within, between]]

        print(dataset)
        df = pingouin.mixed_anova(data=dataset, dv=dependent_variable, subject=subject, within=within, between=between,
                                  effsize=effsize, correction=correction)
        print(df)
        df = df.fillna('')
        all_res = []
        for ind, row in df.iterrows():
            temp_to_append = {
                'id': ind,
                'Source': row['Source'],
                'SS': row['SS'],
                'DF1': row['DF1'],
                'DF2': row['DF2'],
                'MS': row['MS'],
                'F': row['F'],
                'p-unc': row['p-unc'],
                'np2': row[effsize],
                'eps': row['eps'],
            }
            all_res.append(temp_to_append)
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Mixed Anova',
                "test_params": {
                    'selected_depedent_variable': dependent_variable,
                    'selected_subject': subject,
                    'selected_within': within,
                    'selected_between': between,
                    'selected_correction': correction,
                    'selected_effsize': effsize,
                },
                "test_results": all_res,
                "Output_datasets": [],
                'Saved_plots': []
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'status': 'Success', 'Dataframe': all_res},
                            status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        return JSONResponse(content={'status': test_status, 'Dataframe': [], 'col_transormed': '[]'},
                            status_code=200)


@router.get("/calculate_anova_pinguin")
#SS-type should be a valid integer, currently accepting as string in order to use the inlande field validation of strings
async def compute_anova_pinguin(workflow_id: str,
                                 step_id: str,
                                 run_id: str,
                                 dependent_variable: str,
                                 between_factor: list[str] | None = Query(default=None),
                                 ss_type: str | None = Query(2,
                                                             regex="^(1)$|^(2)$|^(3)$"),
                                 effsize: str | None = Query("np2",
                                                             regex="^(np2)$|^(n2)$|^(ng2)$")):

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        selected_datasource = dependent_variable.split("--")[0]
        dependent_variable = dependent_variable.split("--")[1]

        dependent_variable_ping = re.sub("[\(\),:]", " ", dependent_variable)

        between_factor = list(map(lambda x: str(x).split("--")[1], between_factor))

        between_factor_ping = list(map(lambda x: re.sub("[\(\),:]"," ", x), between_factor))

        test_status = 'A column can not be selected multiple times'
        var_list = [dependent_variable] + between_factor
        assert len(var_list) == len(set(var_list))

        test_status = 'Unable to retrieve datasets'
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasource)
        pd.set_option('display.max_columns', None)
        dataset.columns = list(map(lambda x: re.sub("[\(\),:]"," ", x), dataset.columns))


        df = pingouin.anova(data=dataset, dv=dependent_variable_ping, between=between_factor_ping, ss_type=int(ss_type),
                            effsize=effsize, detailed=True)
        print(df)
        df = df.fillna('')
        all_res = []
        for ind, row in df.iterrows():
            temp_to_append = {
                'id': ind,
                'Source': row['Source'],
                'SS': row['SS'],
                'DF': row['DF'],
                'MS': row['MS'],
                'F': row['F'],
                'p-unc': row['p-unc'],
                'np2': row[effsize],
            }
            all_res.append(temp_to_append)
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            file_data = json.load(f)
            file_data['results'] |= {
                "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                "workflow_id": workflow_id,
                "run_id": run_id,
                "step_id": step_id,
                "test_name": 'Anova',
                "test_params": {
                    'selected_depedent_variable': dependent_variable,
                    'selected_between_variables': between_factor,
                    'selected_ss_type': ss_type,
                    'selected_effsize': effsize,
                },
                "test_results": all_res,
                "Output_datasets":[],
                'Saved_plots': []
            }
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()

        print(all_res)
        return JSONResponse(content={'status': 'Success', 'DataFrame': all_res},
                            status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={'status': test_status, 'DataFrame': []},
                            status_code=200)
@router.get("/compute_mean")
async def compute_mean(workflow_id: str,
                                 step_id: str,
                                 run_id: str,
                                 variables: list[str] | None = Query(default=None)):
    df = pd.DataFrame()
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        for ds in selected_datasources:
            dataset = load_data_from_csv(path_to_storage + "/" + ds)
            # Keep requested Columns
            selected_columns = pd.unique(dfv['Variable'])
            for columns in dataset.columns:
                if columns not in selected_columns:
                    dataset = dataset.drop(str(columns), axis=1)
            # Get mean values
            test_status = 'Unable to compute the average values for the selected columns'
            for column in dataset.columns:
                print(str(column))
                print(dataset[str(column)].dtype)
                res = statisticsMean(column, dataset)
                if (res!= -1):
                    df[column] = [res]
                else: df[column] = ["N/A"]
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": 'Mean',
                    "test_params": variables,
                    "test_results":df.to_dict()
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'Dataframe': df.to_json(orient="records")},
                            status_code=200)
    except Exception as e:
        df["Error"] = test_status
        print(e)
        return JSONResponse(content={'Dataframe': df.to_json(orient="records")},
                            status_code=200)

@router.get("/compute_min")
async def compute_min(workflow_id: str,
                                 step_id: str,
                                 run_id: str,
                                 variables: list[str] | None = Query(default=None)):
    df = pd.DataFrame()
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        for ds in selected_datasources:
            dataset = load_data_from_csv(path_to_storage + "/" + ds)
            # Keep requested Columns
            selected_columns = pd.unique(dfv['Variable'])
            for columns in dataset.columns:
                if columns not in selected_columns:
                    dataset = dataset.drop(str(columns), axis=1)
            # Get min values
            test_status = 'Unable to compute the min values for the selected columns'
            for column in dataset.columns:
                res = statisticsMin(column, dataset)
                if (res!= -1):
                    df[column] = [res]
                else: df[column] = ["N/A"]
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": 'Min',
                    "test_params": variables,
                    "test_results":df.to_dict()
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'Dataframe': df.to_json(orient="records")},
                                status_code=200)
    except Exception as e:
        df["Error"] = test_status
        print(e)
        return JSONResponse(content={'Dataframe': df.to_json(orient="records")},
                                status_code=200)

@router.get("/compute_std")
async def compute_std(workflow_id: str,
                       step_id: str,
                       run_id: str,
                       ddof: int | None = Query(default=0),
                       variables: list[str] | None = Query(default=None)):
    df = pd.DataFrame()
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        for ds in selected_datasources:
            dataset = load_data_from_csv(path_to_storage + "/" + ds)
            # Keep requested Columns
            selected_columns = pd.unique(dfv['Variable'])
            for columns in dataset.columns:
                if columns not in selected_columns:
                    dataset = dataset.drop(str(columns), axis=1)
            # Get min values
            test_status = 'Unable to compute the Std for the selected columns'
            for column in dataset.columns:
                res = statisticsStd(column, dataset, ddof)
                if (res!= -1):
                    df[column] = [res]
                else: df[column] = ["N/A"]
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": 'Std',
                    "test_params": variables,
                    "test_results":df.to_dict()
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'Dataframe': df.to_json(orient="records")},
                                status_code=200)
    except Exception as e:
        df["Error"] = test_status
        print(e)
        return JSONResponse(content={'Dataframe': df.to_json(orient="records")},
                                status_code=200)

@router.get("/compute_max")
async def compute_max(workflow_id: str,
                                 step_id: str,
                                 run_id: str,
                                 variables: list[str] | None = Query(default=None)):
    df = pd.DataFrame()
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    # Load Datasets
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))

        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        for ds in selected_datasources:
            dataset = load_data_from_csv(path_to_storage + "/" + ds)
            # Keep requested Columns
            selected_columns = pd.unique(dfv['Variable'])
            for columns in dataset.columns:
                if columns not in selected_columns:
                    dataset = dataset.drop(str(columns), axis=1)
            # Get max values
            test_status = 'Unable to compute the max values for the selected columns'
            for column in dataset.columns:
                res = statisticsMax(column, dataset)
                if (res!= -1):
                    df[column] = [res]
                else: df[column] = ["N/A"]
        test_status = 'Unable to create info.json file'
        with open(path_to_storage + '/output/info.json', 'r+', encoding='utf-8') as f:
            # Load existing data into a dict.
            file_data = json.load(f)
            # Join new data
            new_data = {
                    "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "step_id": step_id,
                    "test_name": 'Max',
                    "test_params": variables,
                    "test_results":df.to_dict()
            }
            file_data['results'] = new_data
            file_data['Output_datasets'] = []
            # Set file's current position at offset.
            f.seek(0)
            # convert back to json.
            json.dump(file_data, f, indent=4)
            f.truncate()
        return JSONResponse(content={'Dataframe': df.to_json(orient="records")},
                            status_code=200)
    except Exception as e:
        df["Error"] = test_status
        print(e)
        return JSONResponse(content={'Dataframe': df.to_json(orient="records")},
                            status_code=200)


@router.get("/fastica")
async def compute_fast_ica(workflow_id: str,
                           step_id: str,
                           run_id: str,
                           n_components: int,
                           max_iter: int | None = Query(default=200),
                           algorithm: str | None = Query("parallel",
                                                         regex="^(parallel)$|^(deflation)$"),
                           fun: str | None = Query("logcosh",
                                                   regex="^(logcosh)$|^(exp)$|^(cube)$"),
                           independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)

    transformer = FastICA(n_components=n_components, max_iter=max_iter, algorithm=algorithm, fun=fun)

    X_transformed = transformer.fit_transform(X)

    df = pd.DataFrame(X_transformed)
    df_components = pd.DataFrame(transformer.components_)
    df_mixing = pd.DataFrame(transformer.mixing_)

    return {'transformed': df.to_json(orient='split'), 'components': df_components.to_json(orient='split'), 'mixing': df_mixing.to_json(orient='split')}

@router.get("/multidimensional_scaling")
async def compute_multidimensional_scaling(workflow_id: str,
                                           step_id: str,
                                           run_id: str,
                                           n_components: int | None = Query(default=2),
                                           max_iter: int | None = Query(default=300),
                                           metric: bool | None = Query(default=True),
                                           dissimilarity: str | None = Query("euclidean",
                                                                             regex="^(euclidean)$|^(precomputed)$"),
                                           independent_variables: list[str] | None = Query(default=None)):

    dataset = load_file_csv_direct(workflow_id, run_id, step_id)

    for columns in dataset.columns:
        if columns not in independent_variables:
            dataset = dataset.drop(str(columns), axis=1)

    X = np.array(dataset)

    transformer = MDS(n_components=n_components, max_iter=max_iter, metric=metric, dissimilarity=dissimilarity)

    X_transformed = transformer.fit_transform(X)

    df = pd.DataFrame(X_transformed)

    df_embedding = pd.DataFrame(transformer.embedding_)
    df_dissimilarity = pd.DataFrame(transformer.dissimilarity_matrix_)

    return {'transformed': df.to_json(orient='split'),
            'position of the dataset in the embedding space': df_embedding.to_json(orient='split'),
            'Pairwise dissimilarities between the points': df_dissimilarity.to_json(orient='split')}

@router.get("/tsne")
async def compute_tsne(workflow_id: str,
                       step_id: str,
                       run_id: str,
                       n_components: int | None = Query(default=2),
                       n_iter: int | None = Query(default=1000),
                       perplexity: float | None = Query(default=30.0),
                       early_exaggeration: float | None = Query(default=12.0),
                       init: str | None = Query("pca",
                                                regex="^(pca)$|^(random)$"),
                       method: str | None = Query("barnes_hut",
                                                  regex="^(barnes_hut)$|^(exact)$"),
                       independent_variables: list[str] | None = Query(default=None)):
    dfv = pd.DataFrame()
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    try:
        test_status = 'Dataset is not defined'
        dfv['variables'] = independent_variables
        dfv[['Datasource', 'Variable']] = dfv["variables"].apply(lambda x: pd.Series(str(x).split("--")))
        independent_variables = dfv["Variable"].tolist()
        selected_datasources = pd.unique(dfv['Datasource'])
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        dataset = load_data_from_csv(path_to_storage + "/" + selected_datasources[0])
        for columns in dataset.columns:
            if columns not in independent_variables:
                dataset = dataset.drop(str(columns), axis=1)
        X = np.array(dataset)


        # perplexity_test = np.arange(5, 100, 5)
        # divergence = []
        #
        # for i in perplexity_test:
        #     model = TSNE(n_components=2, init="pca", perplexity=i)
        #     reduced = model.fit_transform(X)
        #     divergence.append(model.kl_divergence_)
        # fig = px.line(x=perplexity_test, y=divergence, markers=True)
        # fig.update_layout(xaxis_title="Perplexity Values", yaxis_title="Divergence")
        # fig.update_traces(line_color="red", line_width=1)
        # fig.show()


        transformer = TSNE(n_components=n_components, n_iter=n_iter, perplexity=perplexity, early_exaggeration=early_exaggeration, init=init, method=method)
        X_transformed = transformer.fit_transform(X)
        df = pd.DataFrame(X_transformed)
        df_embedding = pd.DataFrame(transformer.embedding_)
        print(transformer.n_features_in_)
        print(transformer.n_iter_)
        print(transformer.square_distances)
        print(transformer.verbose)
        print(transformer.random_state)
        print(transformer.angle)
        print(transformer.metric)

        fig1 = px.scatter(x=X_transformed[:, 0], y=X_transformed[:, 1])
        fig1.update_layout(
            title="t-SNE visualization",
            xaxis_title="First t-SNE",
            yaxis_title="Second t-SNE",
        )
        fig1.show()
        return JSONResponse(content={'status': 'Success',
                                     'transformed': df.to_json(orient='records'), 'embeddings_vector':df_embedding.to_json(orient='records'),
                                     'Kullback_Leibler': transformer.kl_divergence_},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status + e.__str__(),
                                     'transformed': {},
                                     'embeddings_vector': {},
                                     'Kullback_Leibler': ''},
                            status_code=200)

@router.get("/SEM_Optimization")
async def Structural_Equation_Models_Optimization(
        workflow_id: str,
        step_id: str,
        run_id: str,
        file:str,
        model: str,
        obj_func:str):


    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    try:
        if file is None:
            test_status = 'Dataset is not defined'
            raise Exception
        test_status = 'Unable to load the Dataset'
        # We expect only one here
        data = load_data_from_csv(path_to_storage + "/" + file)

    # # TODO: check int64 with floats
    #      because it raises an exception
        print(data.dtypes)
        print('------------------------')
        print('non numeric')
        print(data.select_dtypes(exclude=[np.number]))
        print('int')
        print(data.select_dtypes(include=[np.int]))
        print('float')
        print(data.select_dtypes(include=[np.float]))

        if not data.select_dtypes(include=[np.float]).empty:
            # for num_col in data.dtypes:
            df_num = data.select_dtypes(include=[np.int])
            print(df_num)
            for col in df_num.columns:
                data[col] = pd.to_numeric(data[col], downcast='float')
            print(df_num.dtypes)
            print(data.dtypes)
    # *****************-----------------
    # It's OK up until now
    # *****************-----------------
        test_status = 'Unable to load Model'
        m = Model(model)
        test_status = 'Preparing to fit the model to the data'
        r = m.fit(data, obj=obj_func)
        test_status = 'Unable to calculate inspect parameters estimate'
        ins = m.inspect(std_est=True)
        ins.columns = ins.columns.str.replace('.', '_', regex=True)
        test_status = 'Unable to calculate estimate means'
        means = estimate_means(m)
        cstats = calc_stats(m)
        test_status = 'Unable to calculate factors'
        factors = m.predict_factors(data)
        robust = m.inspect(se_robust=True)
        robust.columns = robust.columns.str.replace('.', '_', regex=True)
        test_status = 'Unable to plot the graph'
        g = semplot(m, filename='t.pdf',plot_covs=True)

        # # TODO: Another implementation of means - We don't use it for now
        # m = ModelMeans(model)
        # m.fit(data)
        # inspect_ModelMeans = m.inspect()
        # inspect_ModelMeans.columns = inspect_ModelMeans.columns.str.replace('.', '_', regex=True)

        # # TODO: FCA
        # # Internet
        # opt = Optimizer(m)
        # objective_function_value = opt.optimize()
        # print('objective_function_value')
        # print(objective_function_value)
        # stats = gather_statistics(opt)
        # print('stats--------')
        # print(type(stats))
        # print(stats)


        return JSONResponse(content={'status': 'Success','fit_results':str(r), 'inspect_means':ins.to_json(orient='records'),'estimate_means':means.to_json(orient='records'),
                                     'factors':factors.to_json(orient='records'),'calc_stats':cstats.to_json(orient='records'),
                                     'robust':robust.to_json(orient='records'),'graph':str(g)},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status+"\n"+e.__str__(),'fit_results':'','inspect_means':'[]','estimate_means':'[]',
                                     'factors':'[]','calc_stats':'[]',
                                     'robust':'[]', 'graph':""},
                            status_code=200)

@router.get("/EFA_extract_latent_structure")
async def Exploratory_Factor_Analysis_extract_latent_structure(
        workflow_id: str,
        step_id: str,
        run_id: str,
        file:str,
        test: str,
        variables: list[str] | None = Query(default=None),
        min_loadings: int | None = Query(default=2),
        pval: float | None = Query(default=0.01),
        levels: int | None = Query(default=2)):

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    dfv = pd.DataFrame()
    try:
        test_status = 'Dataset is not defined'
        if file is None:
            test_status = 'Dataset is not defined'
            raise Exception
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        data = load_data_from_csv(path_to_storage + "/" + file)
        data.columns = data.columns.str.replace("[^a-zA-Z]+", "_", regex=True)
        print(data.columns)
        df = data[data.columns.intersection(variables)]
        print(df.columns)
        print(df.head())
        # TODO: Remove nans
        df = df.dropna()
        print(df.head())
        # data = data.dropna(subset=variables)

        # *****************-------------------------
        # *****************-------------------------
        # if 'Unnamed: 0' in data.columns:
            # data.columns = data.columns.str.replace('.', '_', regex=True)
            # data.columns = data.columns.str.replace(':', '_', regex=True)
            # data.columns = data.columns.str.replace(' ', '_', regex=True)

            # df = data.drop('Unnamed: 0', axis='columns')
        # else:

        print(df.columns)
        dfv1 = pd.DataFrame([df[col].tolist() for col in df.columns], index=df.columns).T
        # print(dfv)
        if test == 'explore_cfa_model':
            test_result = efa.explore_cfa_model(dfv1,min_loadings=min_loadings,pval=pval)
        elif test == 'explore_pine_model':
            test_result = efa.explore_pine_model(dfv1,min_loadings=min_loadings,pval=pval,levels=levels)
        else:
            test_result=''
        print(type(test_result))
        print(test_result)
        # print(pine_test)
        return JSONResponse(content={'status': 'Success', 'test_result': test_result},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status + "\n" + e.__str__(),
                                     'test_result': ''},
                            status_code=200)

@router.get("/Dataframe_preparation")
async def Dataframe_preparation(workflow_id: str,
                                step_id: str,
                                run_id: str,
                                file:str,
                                variables: list[str] | None = Query(default=None),
                                method: str | None = Query("mean",
                                                  regex="^(mean)$|^(median)$|^(most_frequent)$|^(constant)$|^(KNN)$|^(iterative)$")):

    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    # test_status = ''
    # dfv = pd.DataFrame()
    print(path_to_storage)
    try:
        test_status = 'Dataset is not defined'
        if file is None:
            test_status = 'Dataset is not defined'
            raise Exception
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        data = load_data_from_csv(path_to_storage + "/" + file)
        for variable in variables:
            if variable not in data.columns:
                raise Exception(str(variable) + '- The selected variable cannot be found in the dataset.')
        print(method)
        print(variables)
        x = DataframeImputation(data,variables,method)
        if type(x) == str:
            test_status= 'Failed to impute values'
            raise Exception (x)
        return JSONResponse(content={'status': 'Success', 'newdataFrame': x.to_json(orient='records')},
                            status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status + "\n" + e.__str__(),
                             'newdataFrame': '[]'},
                    status_code=200)

@router.get("/hypothesis/autocorrelation", tags=["return_autocorrelation"])
# Validation is done inline in the input of the function
async def hypothesis_autocorrelation(workflow_id: str, step_id: str, run_id: str,
                                 input_name: str,
                                 file:str,
                                 variables: list[str] | None = Query(default=None),
                                 input_adjusted: bool | None = False,
                                 input_qstat: bool | None = False,
                                 input_fft: bool | None = False,
                                 input_bartlett_confint: bool | None = False,
                                 input_missing: str | None = Query("none",
                                                                   regex="^(none)$|^(raise)$|^(conservative)$|^(drop)$"),
                                 input_alpha: float | None = None,
                                 input_nlags: int | None = None,
                                 file_used: str | None = Query("original", regex="^(original)$|^(printed)$")
                                 ) -> dict:
    path_to_storage = get_local_storage_path(workflow_id, run_id, step_id)
    test_status = ''
    dfv = pd.DataFrame()
    try:
        test_status = 'Dataset is not defined'
        if file is None:
            test_status = 'Dataset is not defined'
            raise Exception
        test_status = 'Unable to retrieve datasets'
        # We expect only one here
        data = load_data_from_csv(path_to_storage + "/" + file)
        data.columns = data.columns.str.replace("[^a-zA-Z]+", "_", regex=True)
        print(data.columns)
        df = data[data.columns.intersection(variables)]
        print(df.columns)
        print(df.head())
        # TODO: Remove nans
        df = df.dropna()
        print(df.head())

        z = acf(data, adjusted=input_adjusted, qstat=input_qstat,
                fft=input_fft,
                bartlett_confint=input_bartlett_confint,
                missing=input_missing, alpha=input_alpha,
                nlags=input_nlags)

        to_return = {
            'values_autocorrelation': None,
            'confint': None,
            'qstat': None,
            'pvalues': None
        }

        fig, ax = plt.subplots(nrows=1, ncols=1, facecolor="#F0F0F0")

        ax.legend(["ACF"], loc="upper right", fontsize="x-small", framealpha=1, edgecolor="black", shadow=None)
        ax.grid(which="major", color="grey", linestyle="--", linewidth=0.5)
        print(z[0])

        # Parsing the results of acf into a single object
        # Results will change depending on our input
        if input_qstat and input_alpha:
            to_return['values_autocorrelation'] = z[0].tolist()
            to_return['confint'] = z[1].tolist()
            to_return['qstat'] = z[2].tolist()
            to_return['pvalues'] = z[3].tolist()
            # plot_acf(z, adjusted=input_adjusted, alpha=input_alpha, lags=len(z[0].tolist())-1, ax=ax)
            # ax.set_xticks(np.arange(1, len(z[0].tolist()), step=1))
        elif input_qstat:
            to_return['values_autocorrelation'] = z[0].tolist()
            to_return['qstat'] = z[1].tolist()
            to_return['pvalues'] = z[2].tolist()
            # plot_acf(z, adjusted=input_adjusted, lags=len(z[0].tolist())-1, ax=ax)
            # ax.set_xticks(np.arange(1, len(z[0].tolist()), step=1))
        elif input_alpha:
            to_return['values_autocorrelation'] = z[0].tolist()
            to_return['confint'] = z[1].tolist()
            plot_acf(x=data,
                     adjusted=input_adjusted,
                     # qstat=input_qstat,
                     fft=input_fft,
                     bartlett_confint=input_bartlett_confint,
                     missing=input_missing,
                     alpha=input_alpha,
                     lags=input_nlags,
                     ax=ax,
                     use_vlines=True)
            # plot_acf(z, adjusted=input_adjusted, alpha=input_alpha, lags=len(z[0].tolist()) -1, ax=ax)
            # ax.set_xticks(np.arange(1, len(z[0].tolist()), step=1))
        else:
            to_return['values_autocorrelation'] = z.tolist()
            # plot_acf(z, adjusted=input_adjusted, lags=len(z.tolist())-1, ax=ax)
            # plot_acf(x=raw_data[i], adjusted=input_adjusted, qstat=input_qstat,
            #     fft=input_fft,
            #     bartlett_confint=input_bartlett_confint,
            #     missing=input_missing, alpha=input_alpha,
            #     nlags=input_nlags, ax=ax, use_vlines=True)
            plot_acf(x=data,
                     adjusted=input_adjusted,
                     # qstat=input_qstat,
                     fft=input_fft,
                     bartlett_confint=input_bartlett_confint,
                     missing=input_missing, alpha=input_alpha,
                     ax=ax,
                     lags=input_nlags,
                     use_vlines=True)
                # ax.set_xticks(np.arange(1, len(z.tolist()), step=1))

            # plt.show()
        print("RETURNING VALUES")
        print(to_return)
        plt.savefig(get_local_storage_path(workflow_id, run_id, step_id) + "/output/" + 'autocorrelation.png')

        # plt.show()

        # Prepare the data to be written to the config file
        parameter_data = {
            'name': input_name,
            'adjusted': input_adjusted,
            'qstat': input_qstat,
            'fft': input_fft,
            'bartlett_confint': input_bartlett_confint,
            'missing': input_missing,
            'alpha': input_alpha,
            'nlags': input_nlags,
        }
        result_data = {
            'data_values_autocorrelation': to_return['values_autocorrelation'],
            'data_confint': to_return['confint'],
            'data_qstat': to_return['qstat'],
            'data_pvalues': to_return['pvalues']
        }

        write_function_data_to_config_file(parameter_data, result_data, workflow_id, run_id, step_id)

        return to_return
    except Exception as e:
        print(e)
        return JSONResponse(content={'status': test_status + "\n" + e.__str__(),
                                     'test_result': ''},
                            status_code=200)

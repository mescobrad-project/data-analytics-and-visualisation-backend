import json
import os
from os import walk
from os.path import isfile, join

import requests
from fastapi import APIRouter, Request
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

from app.utils.utils_general import create_local_step

router = APIRouter()

WFAddress = os.environ.get('WFAddress') if os.environ.get(
    'WFAddress') else "https://esbk.platform.mes-cobrad.eu"

TestRunId = os.environ.get('TestRunId') if os.environ.get(
    'TestRunId') else "fe2b0997-6974-4fee-9178-a3291ea1744c"

TestStepId = os.environ.get('TestStepId') if os.environ.get(
    'TestStepId') else "db4a0c7d-0e09-49b2-ba80-7c5e4e3f0768"

NeurodesktopStorageLocation = os.environ.get('NeurodesktopStorageLocation') if os.environ.get(
    'NeurodesktopStorageLocation') else "/neurodesktop-storage"

FrontendAddress = os.environ.get('FrontendAddress') if os.environ.get(
    'FrontendAddress') else "http://localhost:3005"

ExistingFunctions = [
    # EEG
    "auto_correlation",
    "partial_auto_correlation",
    "welch",
    "find_peaks",
    "power_spectral_density_periodogram",
    "stft",
    "power_spectral_density_multitaper",
    "alpha_delta_ratio",
    "predictions",
    "artifacts",
    "alpha_variability",
    "assymetry_indices",
    "slow_waves",
    "spindles",
    "eeg_viewer",
    "eeg_viewer_old",
    "envelop_trend_analysis",
    "sleep_statistic",
    "spectogram_bandpower",
    "slowwave_spindle",
    #  Actigraphy
    "actigraphy_viewer",
    "actigraphy_viewer_general",
    # MRI
    "mri_viewer",
    "free_surfer",
    "recon_all_results",
    "samseg_results",
    # Hypothesis
    "level",
    "normality",
    "normality_anderson",
    "data_transform",
    "pearson_correlation",
    "biweight_midcorrelation",
    "percentage_bend_correlation",
    "shepherd_pi_correlation",
    "skipped_spearman_correlation",
    "point_biserial_correlation",
    "power_spectral_density_main",
    "canonical_correlation",
    'mediation_analysis',
    "data_transform_anova",
    "homoscedasticity",
    "spearman_correlation",
    "kendalltau_correlation",
    "welch_t_test",
    "independent_t_test",
    "t_test_two_samples",
    "mann_whitney_u_rank",
    "wilcoxon_signed_rank",
    "alexander_govern",
    "kruskal_wallis_h",
    "one_way_anova",
    "wilcoxon_rank_statistic",
    "one_way_chi_square",
    "mutliple_comparisons",
    "LDA",
    "SVC",
    "PCA",
    "KMeans",
    "linear_regression",
    "ElasticNet",
    "LassoRegression",
    "RidgeRegression",
    "SGDRegression",
    "HuberRegression",
    "sleep_stage_classification",
    "survivalanalysisriskratiosimple",
    "survivalanalysisriskratiodataset",
    "survivalanalysisriskdifferencesimple",
    'survivalanalysisriskdifferencedataset',
    "survivalanalysisNNTsimple",
    "survivalanalysisNNTdataset",
    "survivalanalysisoddsratiosimple",
    "survivalanalysisoddsratiodataset",
    "survivalanalysisincidencerateratiosimple",
    "survivalanalysisincidencerateratiodataset",
    "survivalanalysisincidenceratedifferencesimple",
    "survivalanalysisincidenceratedifferencedataset",
    "ancova",
    "linearmixedeffectsmodel",
    "survivalanalysiscoxregression",
    "survivalanalysistimevaryingcovariates",
    "survivalanalysiskaplanmeier",
    "principalcomponentanalysis",
    "LinearSVR",
    "LinearSVC",
    "LogisticRegressionPinguin",
    "LogisticRegressionStatsmodels",
    "LogisticRegressionSklearn",
    "fisherexact",
    "mcnemar",
    "LogisticRegressionSklearn",
    "FactorAnalysis",
    "GeneralizedEstimatingEquations",
    "mixed_anova",
    "general_stats_average",
    "general_stats_min",
    "general_stats_max",
    "general_stats_zscore",
    "general_stats_Cov",
    "general_stats_Std",
    'actigraphy_cosinor',
    'actigraphy_metrics',
    "GeneralizedEstimatingEquations",
    "ChooseFactors",
    "GrangerAnalysis",
    "structural_equation_models_optimization",
    'exploratory_factor_analysis_extract_latent_structure',
    # Dashboard
    "dashboard",
]


class FunctionNavigationItem(BaseModel):
    """
    Known metadata information
    "files" : [["bucket_name: "string" , "object_name": "string"]]
     """
    workflow_id: str
    run_id: str
    step_id: str
    function: str
    metadata: dict


# TODO
# @router.get("/test/task/ping", tags=["test_task_ping"])
# async def test_task_ping() -> dict:
#     # channels = data.ch_names
#     print(WFAddress)
#     url = WFAddress + "/run/" + TestRunId + "/step/" + TestStepId + "/ping"
#     print(url)
#     response = requests.get(url)
#     print("Test Response: Task Ping")
#     print(response)
#     return {'test': "test"}

# TODO
# @router.get("/test/task/complete", tags=["test_task_complete"])
# async def test_task_complete(run_id: str,
#                              step_id: str) -> dict:
#     # channels = data.ch_names
#     print(WFAddress)
#     headers = {"Content-Type": "application/json"}
#
#     saved_files = []
#     # saved files should be the same as those upload in a previous step or the call should happen here
#     # TODO
#     data = {
#         "datalake" : saved_files,
#         # "trino:": [        ],
#     }
#     url = WFAddress + "/run/" + run_id + "/step/" + step_id + "/task/script/complete"
#     print(url)
#     response = requests.put(url=url, data=data, headers=headers)
#     print("Test Response: Task Ping")
#     print(response)
#
#     return {'test': response}


@router.get("/task/complete", tags=["test_task_complete"])
async def task_complete(run_id: str,
                             step_id: str) -> dict:
    # channels = data.ch_names
    print(WFAddress)
    headers = {"Content-Type": "application/json"}

    saved_files = []
    data = {
        "data": {
            "datalake": saved_files,
            "trino": []
        }
    }

    url = WFAddress + "/run/" + run_id + "/step/" + step_id + "/task/script/complete"
    print(url)
    response = requests.patch(url=url, data=data, headers=headers)
    print("Test Response: Task Ping")
    print(response)

    return {'test': response}


@router.put("/function/navigation/", tags=["function_navigation"])
async def function_navigation(navigation_item: FunctionNavigationItem) -> dict:
    url_to_redirect = FrontendAddress
    if navigation_item.function:
        match navigation_item.function:
            # EEG
            case "auto_correlation":
                url_to_redirect += "/auto_correlation"
            case "partial_auto_correlation":
                url_to_redirect += "/partial_auto_correlation"
            case "welch":
                url_to_redirect += "/welch"
            case "find_peaks":
                url_to_redirect += "/find_peaks"
            case "power_spectral_density_main":
                url_to_redirect += "/power_spectral_density_main"
            case "power_spectral_density_periodogram":
                url_to_redirect += "/periodogram"
            case "stft":
                url_to_redirect += "/stft"
            case "power_spectral_density_multitaper":
                url_to_redirect += "/power_spectral_density"
            case "alpha_delta_ratio":
                url_to_redirect += "/alpha_delta_ratio"
            case "predictions":
                url_to_redirect += "/predictions"
            case "artifacts":
                url_to_redirect += "/artifacts"
            case "alpha_variability":
                url_to_redirect += "/alpha_variability"
            case "asymmetry_indices":
                url_to_redirect += "/asymmetry_indices"
            case "slow_waves":
                url_to_redirect += "/slowwaves"
            case "spindles":
                url_to_redirect += "/spindles"
            case "sleep_statistic":
                url_to_redirect += "/sleep_statistic"
            case "spectogram_bandpower":
                url_to_redirect += "/spectogram_bandpower"
            case "slowwave_spindle":
                url_to_redirect += "/slowwave_spindle"
            case "sleep_stage_classification":
                url_to_redirect += "/sleep_stage_classification"
            case "manual_sleep_stage_classification":
                url_to_redirect += "/manual_sleep_stage_classification"
            case "eeg_viewer":
                url_to_redirect += "/eeg"
            case "eeg_viewer_old":
                url_to_redirect += "/eeg/old"
            case "envelop_trend_analysis":
                url_to_redirect += "/envelope_trend"
            # case "group_sleep_analysis":
            #     url_to_redirect += "/group_sleep_analysis"
            # case "group_sleep_sensitivity_analysis":
            #     url_to_redirect += "/group_sleep_sensitivity_analysis"
            # case "group_sleep_sensitivity_analysis_add_subject":
            #     url_to_redirect += "/group_sleep_sensitivity_analysis_add_subject"
            # case "group_sleep_sensitivity_analysis_add_subject_final":
            #     url_to_redirect += "/group_sleep_sensitivity_analysis_add_subject_final"
            case "group_common_channels_across_subjects":
                url_to_redirect += "/group_common_channels_across_subjects"
            case "group_sleep_analysis_sensitivity_add_subject_add_channels_final":
                url_to_redirect += "/group_sleep_analysis_sensitivity_add_subject_add_channels_final"
            # Actigraphy
            case "actigraphy_viewer":
                url_to_redirect += "/actigraphy"
            case "actigraphy_viewer_general":
                url_to_redirect += "/actigraphy/general"
            case "actigraphy_page":
                url_to_redirect += "/actigraphy_page"
            case "actigraphy_masking":
                url_to_redirect += "/actigraphy_masking"
            case "actigraphy_analysis":
                url_to_redirect += "/actigraphy_analysis"
            case "actigraphy_cosinor":
                url_to_redirect += "/Actigraphy_Cosinor"
            case "actigraphy_metrics":
                url_to_redirect += "/Actigraphy_Metrics"
            #  MRI
            case "mri_viewer":
                url_to_redirect += "/mri"
            case "free_surfer":
                url_to_redirect += "/freesurfer/recon"
            case "recon_all_results":
                url_to_redirect += "/Freesurfer_ReconAll_Results"
            case "samseg_results":
                url_to_redirect += "/Freesurfer_Samseg_Results"
            # Hypothesis
            case "level":
                url_to_redirect += "/level"
            case "normality":
                url_to_redirect += "/normality_Tests"
            case "normality_anderson":
                url_to_redirect += "/normality_Tests_And"
            case "data_transform":
                url_to_redirect += "/transform_data"
            case "pearson_correlation":
                url_to_redirect += "/Pearson_correlation"
            case "point_biserial_correlation":
                url_to_redirect += "/PointBiserialCorrelation"
            case "data_transform_anova":
                url_to_redirect += "/DataTransformationForANOVA"
            case "homoscedasticity":
                url_to_redirect += "/Homoscedasticity"
            case "spearman_correlation":
                url_to_redirect += "/Spearman_correlation"
            case "kendalltau_correlation":
                url_to_redirect += "/Kendalltau_correlation"
            case "biweight_midcorrelation":
                url_to_redirect += "/Biweight_midcorrelation"
            case "percentage_bend_correlation":
                url_to_redirect += "/Percentage_bend_correlation"
            case "shepherd_pi_correlation":
                url_to_redirect += "/Shepherd_pi_correlation"
            case "skipped_spearman_correlation":
                url_to_redirect += "/Skipped_spearman_correlation"
            case "canonical_correlation":
                url_to_redirect += "/Canonical_correlation"
            case 'mediation_analysis':
                url_to_redirect += "/Mediation_Analysis"
            case "welch_t_test":
                url_to_redirect += "/Welch_t_test"
            case "independent_t_test":
                url_to_redirect += "/Independent_t_test"
            case "t_test_two_samples":
                url_to_redirect += "/Two_Related_samples_t_test"
            case "mann_whitney_u_rank":
                url_to_redirect += "/Mann_Whitney"
            case "wilcoxon_signed_rank":
                url_to_redirect += "/Wilcoxon_signed_rank_test"
            case "alexander_govern":
                url_to_redirect += "/Alexander_Govern_test"
            case "kruskal_wallis_h":
                url_to_redirect += "/Kruskal_Wallis_H_test"
            case "one_way_anova":
                url_to_redirect += "/One_way_ANOVA"
            case "wilcoxon_rank_statistic":
                url_to_redirect += "/Wilcoxon_rank_sum_statistic"
            case "one_way_chi_square":
                url_to_redirect += "/One_way_chi_square_test"
            case "mutliple_comparisons":
                url_to_redirect += "/Multiple_comparisons"
            case "LDA":
                url_to_redirect += "/LDA"
            case "SVC":
                url_to_redirect += "/SVC"
            case "PCA":
                url_to_redirect += "/PCA"
            case "KMeans":
                url_to_redirect += "/KMeans"
            case "linear_regression":
                url_to_redirect += "/linear_regression"
            case "ElasticNet":
                url_to_redirect += "/ElasticNet"
            case "LassoRegression":
                url_to_redirect += "/LassoRegression"
            case "RidgeRegression":
                url_to_redirect += "/RidgeRegression"
            case "SGDRegression":
                url_to_redirect += "/SGDRegression"
            case "HuberRegression":
                url_to_redirect += "/HuberRegression"
            case "LinearSVR":
                url_to_redirect += "/LinearSVR"
            case "LinearSVC":
                url_to_redirect += "/LinearSVC"
            case "LogisticRegressionPinguin":
                url_to_redirect += "/LogisticRegressionPinguin"
            case "LogisticRegressionStatsmodels":
                url_to_redirect += "/LogisticRegressionStatsmodels"
            case "LogisticRegressionSklearn":
                url_to_redirect += "/LogisticRegressionSklearn"
            case "survivalanalysisriskratiosimple":
                url_to_redirect += "/SurvivalAnalysisRiskRatioSimple"
            case "survivalanalysisriskratiodataset":
                url_to_redirect += "/SurvivalAnalysisRiskRatioDataset"
            case "survivalanalysisriskdifferencesimple":
                url_to_redirect += "/SurvivalAnalysisRiskDifferenceSimple"
            case "survivalanalysisriskdifferencedataset":
                url_to_redirect += "/SurvivalAnalysisRiskDifferenceDataset"
            case "survivalanalysisNNTsimple":
                url_to_redirect += "/SurvivalAnalysisNNTSimple"
            case "survivalanalysisNNTdataset":
                url_to_redirect += "/SurvivalAnalysisNNTDataset"
            case "survivalanalysisoddsratiosimple":
                url_to_redirect += "/SurvivalAnalysisOddsRatioSimple"
            case "survivalanalysisoddsratiodataset":
                url_to_redirect += "/SurvivalAnalysisOddsRatioDataset"
            case "survivalanalysisincidencerateratiosimple":
                url_to_redirect += "/SurvivalAnalysisIncidenceRateRatioSimple"
            case "survivalanalysisincidencerateratiodataset":
                url_to_redirect += "/SurvivalAnalysisIncidenceRateRatioDataset"
            case "survivalanalysisincidenceratedifferencesimple":
                url_to_redirect += "/SurvivalAnalysisIncidenceRateDifferenceSimple"
            case "survivalanalysisincidenceratedifferencedataset":
                url_to_redirect += "/SurvivalAnalysisIncidenceRateDifferenceDataset"
            case "survivalanalysiskaplanmeier":
                url_to_redirect += "/SurvivalAnalysisKaplanMeier"
            case "ancova":
                url_to_redirect += "/Ancova"
            case "linearmixedeffectsmodel":
                url_to_redirect += "/LinearMixedEffectsModel"
            case "survivalanalysiscoxregression":
                url_to_redirect += "/SurvivalAnalysisCoxRegression"
            case "survivalanalysistimevaryingcovariates":
                url_to_redirect += "/SurvivalAnalysisTimeVaryingCovariates"
            case "principalcomponentanalysis":
                url_to_redirect += "/PrincipalComponentAnalysis"
            case "fisherexact":
                url_to_redirect += "/FisherExact"
            case "mcnemar":
                url_to_redirect += "/McNemar"
            case "FactorAnalysis":
                url_to_redirect += "/FactorAnalysis"
            case "GeneralizedEstimatingEquations":
                url_to_redirect += "/GeneralizedEstimatingEquations"
            case "mixed_anova":
                url_to_redirect += "/Mixed_Anova"
            case "general_stats_average":
                url_to_redirect += "/General_Stats_Average"
            case "back_average":
                url_to_redirect += "/back_average"
            case "general_stats_min":
                url_to_redirect +="/General_Stats_Min"
            case "general_stats_max":
                url_to_redirect +="/General_Stats_Max"
            case "general_stats_zscore":
                url_to_redirect += "/General_Stats_Zscore"
            case "general_stats_Std":
                url_to_redirect += "/General_Stats_Std"
            case "general_stats_Cov":
                url_to_redirect += "/General_Stats_Cov"
            case "ChooseFactors":
                url_to_redirect += "/ChooseFactors"
            case "GrangerAnalysis":
                url_to_redirect += "/GrangerAnalysis"
            case "PoissonRegression":
                url_to_redirect += "/PoissonRegression"
            case "structural_equation_models_optimization":
                url_to_redirect += "/Structural_Equation_Models_Optimization"
            case "exploratory_factor_analysis_extract_latent_structure":
                url_to_redirect += "/Exploratory_Factor_Analysis_extract_latent_structure"
            # Dashboard
            case "dashboard":
                url_to_redirect += "/dashboard"
        #  Create local storage for files and download them
        # Handle files metadata missing from request/accept it as an empty array
        if "files" in navigation_item.metadata:
            # print("KEY EXISTS")
            # print(navigation_item.metadata)
            create_local_step(workflow_id=navigation_item.workflow_id, run_id=navigation_item.run_id,
                              step_id=navigation_item.step_id, files_to_download=navigation_item.metadata["files"])
        else:
            # print("NOT EXIST KEY")
            # print(navigation_item.metadata)
            create_local_step(workflow_id=navigation_item.workflow_id,
                              run_id=navigation_item.run_id, step_id=navigation_item.step_id, files_to_download=[])

    # Add step and run id to the parameters
    url_to_redirect += "/?run_id="+ navigation_item.run_id+"&step_id=" + navigation_item.step_id + \
                       "&workflow_id=" + navigation_item.workflow_id
    print(url_to_redirect)
    return {"url": url_to_redirect}


@router.get("/function/files/", tags=["function_files"])
async def function_files(workflow_id: str,
                         step_id: str,
                         run_id: str
                         ) -> dict:
    """This function returns the file id needed for a function"""
    files_to_return = [f for f in os.listdir(NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id) if isfile(join(NeurodesktopStorageLocation + '/runtime_config/workflow_' + workflow_id + '/run_' + run_id + '/step_' + step_id, f))]
    return files_to_return

@router.get("/function/existing", tags=["function_existing"], status_code=200)
async def task_existing(request: Request) -> dict:
    return {
        "analytics-functions": ExistingFunctions,
    }

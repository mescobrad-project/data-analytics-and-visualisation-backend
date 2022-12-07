import numpy as np
import pandas as pd
import json
from scipy.stats import jarque_bera, ranksums, chisquare, kruskal, alexandergovern, kendalltau, f_oneway, shapiro, \
    kstest, anderson, normaltest, boxcox, yeojohnson, bartlett, levene, fligner, obrientransform, pearsonr, spearmanr, \
    pointbiserialr, ttest_ind, mannwhitneyu, wilcoxon, ttest_rel, skew, kurtosis, probplot
from typing import Optional, Union, List
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import matplotlib.pyplot as plt
import mpld3
from fastapi import FastAPI, Path, Query, APIRouter
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from app.pydantic_models import ModelMultipleComparisons
from app.utils.utils_datalake import fget_object, get_saved_dataset_for_Hypothesis
from app.utils.utils_general import get_local_storage_path, get_single_file_from_local_temp_storage, load_data_from_csv, \
    load_file_csv_direct
import scipy.stats as st
import statistics
from tabulate import tabulate

router = APIRouter()
data = pd.read_csv('example_data/sample_questionnaire.csv')

def normality_test_content_results(column: str):
    if (column):
        # region Creating Box-plot
        fig2 = plt.figure()
        plt.boxplot(data[str(column)])
        plt.ylabel("", fontsize=14)
        # show plot
        plt.show()
        html_str_B = mpld3.fig_to_html(fig2)
        #endregion
        # region Creating QQ-plot
        fig = sm.qqplot(data[str(column)], line='45')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        html_str = mpld3.fig_to_html(fig)
        # endregion
        # region Creating Probability-plot
        fig3 = plt.figure()
        ax1 = fig3.add_subplot()
        prob =  probplot(data[str(column)], dist=st.norm, plot=ax1)
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
        axs.hist(data[str(column)], density=True, bins=30, label="Data", rwidth=0.9,
                 color='#607c8e')

        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 300)
        kde = st.gaussian_kde(data[str(column)])
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
        skewtosend = skew(data[str(column)], axis=0, bias=True)
        kurtosistosend = kurtosis(data[str(column)], axis=0, bias=True)
        st_dev = np.std(data[str(column)])
        # Used Statistics lib for cross-checking
        # standard_deviation = statistics.stdev(data[str(column)])
        median_value = float(np.percentile(data[str(column)], 50))
        # Used a different way to calculate Median
        # TODO: we must investigate why it returns a different value
        # med2 = np.median(data[str(column)])
        mean_value = np.mean(data[str(column)])
        num_rows = data[str(column)].shape
        top5 = sorted(data[str(column)].tolist(), reverse=True)[:5]
        last5 = sorted(data[str(column)].tolist(), reverse=True)[-5:]
        #endregion
        return {'qqplot': html_str, 'histogramplot': html_str_H, 'boxplot': html_str_B, 'probplot': html_str_P, 'skew': skewtosend, 'kurtosis': kurtosistosend, 'standard_deviation': st_dev, "median": median_value, "mean": mean_value, "sample_N": num_rows, "top_5": top5, "last_5": last5}
    else:
        return {'qqplot': "", 'histogramplot': "", 'boxplot': "", 'probplot': "",
                'skew': 0, 'kurtosis': 0,
                'standard_deviation': 0, "median": 0,
                "mean": 0, "sample_N": 0, "top_5": [], "last_5": []}


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
                                                   regex="^(Shapiro-Wilk)$|^(Kolmogorov-Smirnov)$|^(Anderson-Darling)$|^(D’Agostino’s K\^2)$|(Jarque-Bera)$")) -> dict:

    data = load_file_csv_direct(run_id, step_id)
    results_to_send = normality_test_content_results(column)

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
                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
                list_anderson.append('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
            else:
                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
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
        print(pvalue)
        if pvalue > 0.05:
            return {'statistic': statistic, 'p_value': pvalue,
                    'Description': 'Sample looks Gaussian (fail to reject H0)', 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}
        else:
            return {'statistic': statistic, 'p_value': pvalue,
                    'Description': 'Sample does not look Gaussian (reject H0)', 'data': tabulate(data, headers='keys', tablefmt='html'), 'results': results_to_send}


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
                boxcox_array, maxlog, z = boxcox(np.array(data[str(column)]), alpha=alpha)
                return {'Box-Cox power transformed array': list(boxcox_array), 'lambda that maximizes the log-likelihood function': maxlog, 'minimum confidence limit': z[0], 'maximum confidence limit': z[1]}
        else:
            if alpha == None:
                y = boxcox(np.array(data[str(column)]), lmbda=lmbd)
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

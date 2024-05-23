import matplotlib.pyplot as plt
import mpld3
import pandas as pd
import pingouin
import plotly
from lifelines.fitters.npmle import min_max
from scipy.stats import probplot, skew, kurtosis, sem, t
import scipy.stats as st
import numpy as np
import math
import statistics
from factor_analyzer.utils import cov
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns

def create_plots(plot_type: str, column: str, second_column: str, selected_dataframe, path_to_storage:str, filename:str):
    if plot_type == 'BoxPlot':
        try:
            fig1 = go.Figure()
            fig1.add_trace(go.Box(
                y=selected_dataframe[str(column)],
                name="Suspected Outliers",
                boxpoints='suspectedoutliers',  # only suspected outliers
                marker=dict(
                    color='rgb(8,81,156)',
                    outliercolor='rgba(219, 64, 82, 0.6)',
                    line=dict(
                        outliercolor='rgba(219, 64, 82, 0.6)',
                        outlierwidth=2)),
                line_color='rgb(8,81,156)'
            ))
            # fig1.update_layout(title_text="Box Plot Styling Outliers")

            # fig, ax1 = plt.subplots()
            # ax1.set_title('Box Plot')
            # if len(selected_dataframe) == 2:
            #     # TODO: For Biserial, but in an other case if it finds 2 it will do the same
            #     plt.boxplot(selected_dataframe, positions=[0, 1])
            #     plt.ylabel(column, fontsize=14)
            #     plt.xlabel(second_column, fontsize=14)
            # else:
            #     plt.boxplot(selected_dataframe[str(column)])
            # plt.ylabel("", fontsize=14)
            # plt.xticks(fontsize=12)
            # plt.yticks(fontsize=12)
            # plt.savefig(path_to_storage + "/output/"+filename+".svg", format="svg")
            # plt.show()
            # fig1.show()
            fig1.write_image(path_to_storage + "/output/" + filename + ".svg")
            html_str = plotly.io.to_json(fig1, validate=True, pretty=True, remove_uids=True, engine=None)
            # html_str = mpld3.fig_to_html(fig)
            return html_str
        except Exception as e:
            print(e)
            print("Error : Creating BoxPlot \n"+e.__str__())
            return {"Error : Creating BoxPlot", e}
    if plot_type == "QQPlot":
        try:
            # fig = plt.figure()
            # ax = fig.add_subplot()
            # pingouin.qqplot(selected_dataframe[str(column)], dist='norm', ax=ax)
            # # We changed to Pingouin because it's better
            # # fig = sm.qqplot(selected_dataframe[str(column)], line='45')
            # plt.xticks(fontsize=12)
            # plt.yticks(fontsize=12)
            # plt.savefig(path_to_storage + "/output/"+filename+".svg", format="svg")
            # plt.show()
            # html_str = mpld3.fig_to_html(fig)

            qqplot_data = qqplot(selected_dataframe[str(column)], line='s').gca().lines
            fig = go.Figure()
            fig.add_trace({
                'type': 'scatter',
                'x': qqplot_data[0].get_xdata(),
                'y': qqplot_data[0].get_ydata(),
                'mode': 'markers',
                'marker': {
                    'color': '#19d3f3'
                }
            })
            fig.add_trace({
                'type': 'scatter',
                'x': qqplot_data[1].get_xdata(),
                'y': qqplot_data[1].get_ydata(),
                'mode': 'lines',
                'line': {
                    'color': '#636efa'
                }

            })
            fig['layout'].update({
                'title': 'Quantile-Quantile Plot',
                'xaxis': {
                    'title': 'Theoritical Quantities',
                    'zeroline': False
                },
                'yaxis': {
                    'title': 'Sample Quantities'
                },
                'showlegend': False,
                # 'width': 800,
                # 'height': 700,
            })

            # py.iplot(fig, filename='normality-QQ')
            fig.write_image(path_to_storage + "/output/" + filename + ".svg")
            html_str = plotly.io.to_json(fig, validate=True, pretty=True, remove_uids=True, engine=None)

            return html_str
        except Exception as e:
            print(e)
            print("Error : Creating QQPlot \n"+e.__str__())
            return {}
    if plot_type == 'PPlot':
        try:
            fig = plt.figure()
            ax1 = fig.add_subplot()
            prob = probplot(selected_dataframe[str(column)], dist=st.norm, plot=ax1)
            ax1.set_title('Probplot against normal distribution')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.savefig(path_to_storage + "/output/"+filename+".svg", format="svg")
            plt.show()
            html_str = mpld3.fig_to_html(fig)
            return html_str
        except Exception as e:
            print(e)
            print("Error : Creating PPlot \n"+e.__str__())
            return {}
    if plot_type == 'HistogramPlot':
        try:
            # fig, axs = plt.subplots(1, 1,
            #                          # figsize=(640, 480),
            #                          tight_layout=True)
            #
            # ## q25, q75 = np.percentile(data[str(column)], [25, 75])
            # ## bin_width = 2 * (q75 - q25) * len(data[str(column)]) ** (-1 / 3)
            # ## bins = round((data[str(column)].max() - data[str(column)].min()) / bin_width)
            # axs.hist(selected_dataframe[str(column)], density=True, bins=30, label="Data", rwidth=0.9,
            #          color='#607c8e')
            # mn, mx = plt.xlim()
            # plt.xlim(mn, mx)
            # kde_xs = np.linspace(mn, mx, 300)
            # kde = st.gaussian_kde(selected_dataframe[str(column)])
            # plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
            # plt.legend(loc="upper left")
            # plt.ylabel("Probability", fontsize=14)
            # plt.xlabel("Data", fontsize=14)
            # plt.title("Histogram", fontsize=16)
            # plt.xticks(fontsize=12)
            # plt.yticks(fontsize=12)
            # plt.savefig(path_to_storage + "/output/"+filename+".svg", format="svg")
            # plt.show()

            # remove nans
            # df = selected_dataframe[str(column)].apply(lambda x: pd.Series(x.dropna().values))
            df = selected_dataframe[str(column)].dropna()
            # create figure
            fig = ff.create_distplot(hist_data=[df], group_labels=[column],curve_type='kde',bin_size=0.5,histnorm='probability')
            # fig = go.Figure(data=[go.Histogram(x=selected_dataframe[str(column)],histnorm='probability',bingroup=0.5)])
            # fig.show()
            fig.write_image(path_to_storage + "/output/"+filename+".svg")
            html_str = fig.to_json(pretty=True)
            # html_str = mpld3.fig_to_html(fig)
            return html_str
        except Exception as e:
            print(e)
            print("Error : Creating HistogramPlot \n"+e.__str__())
            return {}
    if plot_type == 'Scatter_Two_Variables':
        try:
            fig = plt.figure()
            ax1 = fig.add_subplot()
            prob = plt.scatter(selected_dataframe[str(column)], selected_dataframe[str(second_column)],
                    color='blue', marker="*")
            ax1.set_title('Selected variables plot')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylabel(second_column, fontsize=14)
            plt.xlabel(column, fontsize=14)
            plt.xticks(np.arange(min(selected_dataframe[str(column)]), max(selected_dataframe[str(column)])+1, 1.0))
            plt.savefig(path_to_storage + "/output/"+filename+".svg", format="svg")
            plt.show()
            html_str = mpld3.fig_to_html(fig)
            return html_str
        except Exception as e:
            print(e)
            print("Error : Creating Scatter Plot \n"+e.__str__())
            return {}
    else:
        return -1


def compute_skewness(column: str, selected_dataframe):
    try:
        # We use bias=False to comply with their calculation
        # result = skew(selected_dataframe[str(column)], axis=0, bias=True)
        result = skew(selected_dataframe[str(column)], axis=0, bias=False)
        # print('stats.skew(bias=false)')
        # print(skew(selected_dataframe[str(column)], axis=0, bias=False))
        # print('df.skew()')
        # print(selected_dataframe[str(column)].skew())
        if math.isnan(result):
            result = ''
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute skew \n"+e.__str__())
        return {}


def compute_kurtosis(column: str, selected_dataframe):
    try:
        # We use bias=False to comply with their calculation
        # result = kurtosis(selected_dataframe[str(column)], axis=0, bias=True)
        result = kurtosis(selected_dataframe[str(column)], axis=0, bias=False)
        # print('stats.kurtosis(bias=false)')
        # print(kurtosis(selected_dataframe[str(column)], axis=0, bias=False))
        # print('df.kurtosis()')
        # print(selected_dataframe[str(column)].kurtosis())
        if math.isnan(result):
            result = ''
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute kurtosis \n"+e.__str__())
        return {}


def outliers_removal(column: str, selected_dataframe):
    try:
        for x in [column]:
            q75, q25 = np.percentile(selected_dataframe.loc[:, x], [75, 25])
            intr_qr = q75 - q25
            max = q75 + (1.5 * intr_qr)
            min = q25 - (1.5 * intr_qr)
            outliers = selected_dataframe[(selected_dataframe[x] < min) | (selected_dataframe[x] > max)]
            selected_dataframe = selected_dataframe[(selected_dataframe[x] >= min) & (selected_dataframe[x] <= max)]
            # selected_dataframe.loc[selected_dataframe[x] < min, x] = np.nan
            # selected_dataframe.loc[selected_dataframe[x] > max, x] = np.nan
            # selected_dataframe = selected_dataframe.dropna(axis=0)
            return selected_dataframe, outliers
    except Exception as e:
        print(e)
        print("Error : Failed to remove outliers \n"+e.__str__())
        return {}

def statisticsMean(column: str, selected_dataframe):
    try:
        df2 = selected_dataframe.dropna(subset=[str(column)])
        result = statistics.mean(df2[str(column)])
        if math.isnan(result):
            result = ''
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute Mean for column: "+column+"\n"+e.__str__())
        return -1

def statisticsMin(column: str, selected_dataframe):
    try:
        df2 = selected_dataframe.dropna(subset=[str(column)])
        if pd.to_numeric(df2[str(column)], errors='coerce').notnull().all():
            result = min(df2[str(column)])
        else:
            raise Exception
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute Min for column: "+column+"\n"+e.__str__())
        return -1

def statisticsMax(column: str, selected_dataframe):
    try:
        df2 = selected_dataframe.dropna(subset=[str(column)])
        if pd.to_numeric(df2[str(column)], errors='coerce').notnull().all():
            result = max(df2[str(column)])
        else:
            raise Exception
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute Max for column: "+column+"\n"+e.__str__())
        return -1

def statisticsStd(column: str, selected_dataframe, ddof):
    try:
        # df2 = selected_dataframe.dropna(subset=[str(column)])
        result = np.std(selected_dataframe[str(column)], ddof=ddof)
        # The result is the same with ddof=1
        # print('df.std()')
        # print(selected_dataframe[str(column)].std())
        if math.isnan(result):
            result = ''
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute Std for column: "+column +"\n"+e.__str__())
        return -1

def statisticsCov(selected_dataframe, ddof):
    try:
        result = cov(selected_dataframe, ddof=ddof)
        result = [
            tuple('' if isinstance(i, float) and math.isnan(i) else i for i in t)
            for t in result
        ]
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute Covariance Matrix \n"+e.__str__())
        return ()

def statisticsVar(column: str, selected_dataframe, ddof = 1):
    '''ddof=1 for Sample variance end 0 for population variance'''
    try:
        result = selected_dataframe[str(column)].var(ddof=ddof)
        if math.isnan(result):
            result = ''
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute Variance for column: "+column +"\n"+e.__str__())
        return -1

def statisticsStandardError(column: str, selected_dataframe, ddof = 1):
    try:
        result = sem(selected_dataframe[column])
        if math.isnan(result):
            result = ''
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute Variance for column: "+column +"\n"+e.__str__())
        return -1

def statisticsConfidenceLevel(column: str, selected_dataframe, alpha = 0.95):
    try:
        l,m = t.interval(alpha=alpha, df=len(selected_dataframe[column]) - 1, loc=np.mean(selected_dataframe[column]),
                       scale=sem(selected_dataframe[column]))
        result = (m-l)/2
        if math.isnan(result):
            result = ''
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute Variance for column: "+column +"\n"+e.__str__())
        return -1

def DataframeImputation(selected_dataframe, selected_variables, method):
    try:
        df = selected_dataframe
        print(df.dtypes)
        print(df.shape)
        NA = pd.DataFrame(data=[df.isna().sum().tolist(), ["{:.2f}".format(i) + '%' \
                                                           for i in (df.isna().sum() / df.shape[0] * 100).tolist()]],
                          columns=df.columns, index=['NA Count', 'NA Percent']).transpose()
        NA.style.background_gradient(cmap="Pastel1_r", subset=['NA Count'])
        print(NA)
        data1 = df.copy()
        data2 = df.copy()
        print(selected_variables)
        if method == 'mean' or method == 'median' or method == 'iterative':
            print(method)
            for variable in selected_variables:
                print(df[variable].dtype)
                if df[variable].dtype == object:
                    raise Exception('Mean or Median or Iterative methods can only be used with numeric data.')

        if method == 'mean':
            imp = SimpleImputer(strategy='mean')
            data1[selected_variables] = imp.fit_transform(data1[selected_variables])
            # data1[selected_variables] = imp.fit_transform(data1[selected_variables].values.reshape(-1, 1))
        elif method == 'median':
            imp = SimpleImputer(strategy='median')
            data1[selected_variables] = imp.fit_transform(data1[selected_variables])
        elif method == 'most_frequent':
            imp = SimpleImputer(strategy='most_frequent')
            data1[selected_variables] = imp.fit_transform(data1[selected_variables])
        elif method == 'constant':
            imp = SimpleImputer(strategy='constant')
            data1[selected_variables] = imp.fit_transform(data1[selected_variables])
        elif method == 'KNN':
            print('KNN')
            # data1[selected_variable].replace('NaN', np.nan)
            # imputer = KNNImputer(n_neighbors=2, missing_values=np.nan)
            imputer = KNNImputer(n_neighbors=5)
            data1[selected_variables] = imputer.fit_transform(data1[selected_variables])
        elif method == 'iterative':
            imp = IterativeImputer(max_iter=10, random_state=0)
            data1[selected_variables] = imp.fit_transform(data1[selected_variables])
        # After
        NA = pd.DataFrame(data=[data1.isna().sum().tolist(), ["{:.2f}".format(i) + '%' \
                                                           for i in (data1.isna().sum() / data1.shape[0] * 100).tolist()]],
                          columns=data1.columns, index=['NA Count', 'NA Percent']).transpose()
        NA.style.background_gradient(cmap="Pastel1_r", subset=['NA Count'])
        print(NA)
        return data1
    except Exception as e:
        print("Error : Failed to impute values: " + "\n" + e.__str__())
        return "Error : " + "\n" + e.__str__()

import matplotlib.pyplot as plt
import mpld3
import pandas as pd
import pingouin
from lifelines.fitters.npmle import min_max
from scipy.stats import probplot, skew, kurtosis
import scipy.stats as st
import numpy as np
import statistics


def create_plots(plot_type: str, column: str, second_column: str, selected_dataframe, path_to_storage:str, filename:str):
    if plot_type == 'BoxPlot':
        try:
            fig, ax1 = plt.subplots()
            ax1.set_title('Box Plot')
            if len(selected_dataframe) == 2:
                # TODO: For Biserial, but in an other case if it finds 2 it will do the same
                plt.boxplot(selected_dataframe, positions=[0, 1])
                plt.ylabel(column, fontsize=14)
                plt.xlabel(second_column, fontsize=14)
            else:
                plt.boxplot(selected_dataframe[str(column)])
            plt.ylabel("", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.savefig(path_to_storage + "/output/"+filename+".svg", format="svg")
            plt.show()
            html_str = mpld3.fig_to_html(fig)
            return html_str
        except Exception as e:
            print(e)
            print("Error : Creating BoxPlot")
            return {"Error : Creating BoxPlot", e}
    if plot_type == "QQPlot":
        try:
            fig = plt.figure()
            ax = fig.add_subplot()
            pingouin.qqplot(selected_dataframe[str(column)], dist='norm', ax=ax)
            # We changed to Pingouin because it's better
            # fig = sm.qqplot(selected_dataframe[str(column)], line='45')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.savefig(path_to_storage + "/output/"+filename+".svg", format="svg")
            plt.show()
            html_str = mpld3.fig_to_html(fig)
            return html_str
        except Exception as e:
            print(e)
            print("Error : Creating QQPlot")
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
            print("Error : Creating PPlot")
            return {}
    if plot_type == 'HistogramPlot':
        try:
            fig, axs = plt.subplots(1, 1,
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
            plt.savefig(path_to_storage + "/output/"+filename+".svg", format="svg")
            plt.show()
            html_str = mpld3.fig_to_html(fig)
            return html_str
        except Exception as e:
            print(e)
            print("Error : Creating HistogramPlot")
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
            print("Error : Creating Scatter Plot")
            return {}
    else:
        return -1


def compute_skewness(column: str, selected_dataframe):
    try:
        result = skew(selected_dataframe[str(column)], axis=0, bias=True)
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute skew")
        return {}


def compute_kurtosis(column: str, selected_dataframe):
    try:
        result = kurtosis(selected_dataframe[str(column)], axis=0, bias=True)
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute kurtosis")
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
        print("Error : Failed to remove outliers")
        return {}

def statisticsMean(column: str, selected_dataframe):
    try:
        df2 = selected_dataframe.dropna(subset=[str(column)])
        result = statistics.mean(df2[str(column)])
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute Mean for column: "+column)
        return -1

def statisticsMin(column: str, selected_dataframe):
    try:
        df2 = selected_dataframe.dropna(subset=[str(column)])
        if pd.to_numeric(df2[str(column)], errors='coerce').notnull().all():
            result = max(df2[str(column)])
        else:
            raise Exception
        return result
    except Exception as e:
        print(e)
        print("Error : Failed to compute Min for column: "+column)
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
        print("Error : Failed to compute Max for column: "+column)
        return -1


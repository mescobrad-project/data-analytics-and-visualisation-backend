import matplotlib.pyplot as plt
import mpld3
import pingouin
from scipy.stats import probplot, skew
import scipy.stats as st
import numpy as np

def create_plots(plot_type: str, column: str, selected_dataframe):
    if plot_type == 'BoxPlot':
        try:
            fig = plt.figure()
            plt.boxplot(selected_dataframe[str(column)])
            plt.ylabel("", fontsize=14)
            plt.show()
            html_str = mpld3.fig_to_html(fig)
            return html_str
        except Exception as e:
            print(e)
            print("Error : Creating BoxPlot function")
            return {}
    if plot_type == "QQPlot":
        try:
            fig = plt.figure()
            ax = fig.add_subplot()
            pingouin.qqplot(selected_dataframe[str(column)], dist='norm', ax=ax)
            # We changed to Pingouin because it's better
            # fig = sm.qqplot(selected_dataframe[str(column)], line='45')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.show()
            html_str = mpld3.fig_to_html(fig)
            return html_str
        except Exception as e:
            print(e)
            print("Error : Creating BoxPlot function")
            return {}
    if plot_type == 'PPlot':
        try:
            fig = plt.figure()
            ax1 = fig.add_subplot()
            prob = probplot(selected_dataframe[str(column)], dist=st.norm, plot=ax1)
            ax1.set_title('Probplot against normal distribution')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.show()
            html_str = mpld3.fig_to_html(fig)
            return html_str
        except Exception as e:
            print(e)
            print("Error : Creating BoxPlot function")
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
            plt.show()
            html_str = mpld3.fig_to_html(fig)
            return html_str
        except Exception as e:
            print(e)
            print("Error : Creating BoxPlot function")
            return {}
    else:
        return -1

def compute_skewness(column: str, selected_dataframe, args):
    try:
        result = skew(selected_dataframe[str(column)], *args)
        return result
    except Exception as e:
        print(e)
        print("Error : Creating BoxPlot function")
        return {}

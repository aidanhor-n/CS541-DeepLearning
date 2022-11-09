import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


def visualize_pdf(data1, data2, feature="Ellipticity", name1="Pure", name2="Mixture"):
    """
    create a shaded normalized probability density function graph of the two datasets
    :param data1: list of feature values
    :param data2: list of feature values
    :param feature: name of feature
    :param name1: name or type of powder 1
    :param name2: name or type of powder 2
    :return: a graph
    """

    # If you want to use a histogram instead, use the following:
    # plt.hist(data1, density=True, label=name1, fc=(0, 0, 1, 0.5))
    # plt.hist(data2, density=True, label=name2, fc=(1, 0, 0, 0.5))

    # get the x and y data of the PDF
    ax1 = get_pdf_xy(data1, name1)
    ax2 = get_pdf_xy(data2, name2)

    plt.fill_between(ax1[0], ax1[1], 0, facecolor="darkorange", color="darkorange", alpha=0.2)
    plt.plot(ax1[0], ax1[1], color='orange', linewidth=3)
    plt.fill_between(ax2[0], ax2[1], 0, facecolor="cornflowerblue", color="cornflowerblue", alpha=0.2)
    plt.plot(ax2[0], ax2[1], color='cornflowerblue', linewidth=3)

    plt.legend(labels=[name1, name2])
    plt.ylabel("Probability")
    plt.xlabel(feature)
    plt.title("Probability Density Function: " + feature)

    plt.show()


def get_pdf_xy(data, name):
    """
    creates a probability density function of the given dataset and returns its x-values and normalized y-values
    https://stackoverflow.com/questions/55128462/how-to-normalize-seaborn-distplot
    :param data: the dataset to be plotted
    :param name: the name of the data
    :return: the x values of the dataset, the normalized y values of the dataset
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # create the plot of the data (which we are going to normalize)
    g = sns.distplot(data, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3},
                     label=name, norm_hist=True, ax=ax)

    # now get the data and normalize it
    line = g.get_lines()[0]
    xd = line.get_xdata()
    yd = line.get_ydata()
    normed_yd = normalize(yd)

    # need to close the plot so it doesn't show up in Jupyter
    plt.close()

    return xd, normed_yd


def normalize(x):
    """
    normalizes the given data in x
    https://stackoverflow.com/questions/29661574/normalize-numpy-array-columns-in-python
    :param x: a list of data points to be normalized
    :return: the normalized data from x
    """
    return(x - x.min(0)) / x.ptp(0)


def compare_pdf(data1, data2, a=0.05):
    """
    Tests the null hypothesis that 2 samples are drawn from the same distribution.
    If the p-value is high,
    then we cannot reject the hypothesis that the distributions of the two samples are the same.
    If the size of data1 and data2 are equal, you can reject regardless of the KS statistic observed
    :param data1: list of feature values
    :param data2: list of feature values
    :param a: The threshold p-value for significance (defaults to 0.05)
    :return Two-tailed P-value, the Ks statistic, and if the statistic is similar
    """
    ks_statistic, p_value = stats.ks_2samp(data1, data2, alternative="two-sided", mode="auto")
    if p_value < a:
        different = True
    else:
        different = False
    return {"ks": ks_statistic, "p": p_value, "different": different}


def generate_bi_modal_data():
    """
    generate a bi-modal test dataset
    :return: list of datapoints
    """
    n = 1000
    mu, sigma = 0.3, .07
    mu2, sigma2 = 0.8, .04
    x1 = np.random.normal(mu, sigma, n)
    x2 = np.random.normal(mu2, sigma2, n)
    x = np.concatenate([x1, x2])
    return x


def generate_normal_data():
    """
    generate a normal distribution test dataset
    :return: list of datapoints
    """
    n = 1000
    mu, sigma = 0.5, .1
    x = np.random.normal(mu, sigma, n)
    return x

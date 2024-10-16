import codecs
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from cycler import cycler
from lmfit.models import SkewedGaussianModel
from scipy.interpolate import interp1d


def read_delta_file(file_path):
    accs = []
    with codecs.open(file_path, encoding='utf-8', mode='r') as ifile:
        for line in ifile:
            if line.strip() == '':
                continue
            acc = float(line.strip())
            accs.append(acc)
    return accs


def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))


def fit_gaussian(bin_borders, bin_heights, pps, bin_nb):
    # https://stackoverflow.com/questions/72908352/curve-does-not-fit-with-the-histogram-for-customize-function
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

    x0 = [max(bin_heights), np.mean(pps), np.std(pps)]

    # first try
    popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, x0)
    a, b, c = popt

    popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=[a, b, c])

    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], bin_nb)

    return x_interval_for_fit, popt


def fit_skewed_gaussian(bin_borders, bin_heights, pps, bin_nb):
    # https://stackoverflow.com/questions/25903487/fitting-a-histogram-with-skewed-gaussian
    # https://lmfit.github.io/lmfit-py/builtin_models.html
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

    model = SkewedGaussianModel()

    params = model.guess(bin_heights, x=bin_centers)
    result = model.fit(bin_heights, params, x=bin_centers)

    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], bin_nb)

    #return x_interval_for_fit, result.best_values['amplitude'], result.best_values['center'], result.best_values['sigma'], result.best_values['gamma']
    return x_interval_for_fit, result


def plot_histogram_fit(axs, data, label, bin_nb, bin_edges):
    # Plot perplexities histogram
    if bin_edges is None:
        bin_heights, bin_borders, _ = axs.hist(data, bins=bin_nb, label=label, alpha=0.3)
    else:
        # Use the same bins as previous plot
        bin_heights, bin_borders, _ = axs.hist(data, bins=bin_edges, label=label, alpha=0.3)

    """
    # Fit a Gaussian to the histogram
    x_interval_for_fit, popt = fit_gaussian(bin_borders, bin_heights, data, bin_nb)

    # Interpolate to smooth curves
    # https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
    f_interp = interp1d(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), kind='cubic')

    # More points for smoother curve
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], bin_nb * 10)

    # Plot the Gaussian fit
    #axs.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), label=f'fit')
    axs.plot(x_interval_for_fit, f_interp(x_interval_for_fit), label=f'fit')
    """

    # Fit a Skewed Gaussian to the histogram
    x_interval_for_fit, result = fit_skewed_gaussian(bin_borders, bin_heights, data, bin_nb)

    # Interpolate to smooth curves
    # https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
    f_interp = interp1d(x_interval_for_fit, result.best_fit, kind='cubic')

    # More points for smoother curve
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], bin_nb * 10)

    # Plot the Skewed Gaussian fit
    # axs.plot(x_interval_for_fit, result.best_fit, label=f'skewed fit')
    axs.plot(x_interval_for_fit, f_interp(x_interval_for_fit), label=f'skewed fit {label}')

    return bin_borders


def generate_file_paths(base_file_name, test, model, condition, items, item_type):
    if item_type == 'conditions':
        return [f'{acc_file_dir}{test}{base_file_name}.{model}.{item}' for item in items]
    elif item_type == 'tests':
        return [f'{acc_file_dir}{item}{base_file_name}.{model}.{condition}' for item in items]
    elif item_type == 'models':
        return [f'{acc_file_dir}{test}{base_file_name}.{item}.{condition}' for item in items]
    else:
        raise ValueError(f'Unknown item_type: {item_type}')


def harmonize_bins(file_paths, bin_nb):
    biggest_int = 0
    dict_interval = {}
    for file_path in file_paths:
        pps = read_delta_file(file_path)
        print(f'{min(pps)}, {max(pps)}')
        interval = abs(max(pps) - min(pps))
        print(interval)
        dict_interval[interval] = pps

    biggest_int = max(dict_interval.keys())

    pps_big_interval = dict_interval[biggest_int]
    _, bin_borders = np.histogram(pps_big_interval, bin_nb)

    return bin_borders


def compare_pps(file_paths):
    pps_list = []
    for file_path in file_paths:
        pps_list.append(read_delta_file(file_path))

    higher = 0
    for i in range(len(pps_list[0])):
        if pps_list[0][i] < pps_list[1][i]:
            higher += 1

    print(f'First is lower perplexity in: {higher/len(pps_list[0]) * 100}% of cases')


def plot_from_files(ax, test_name, model, condition, items, bin_nb, labels, item_type):

    base_file_name = '_semcor_wordnet_aa_n_5000_rand_160.csv.curated.200.csv'

    file_paths = generate_file_paths(base_file_name,
                                     test_name,
                                     model,
                                     condition,
                                     items,
                                     item_type)
    print(file_paths)

    if item_type == 'conditions':
        title = f'{test_name}, {model}'
    elif item_type == 'tests':
        title = f"{model.split('.')[1]}"
    else:
        title = f'{test_name}, {condition}'

    bin_edges = harmonize_bins(file_paths, bin_nb)

    for file_path, label in zip(file_paths, labels):
        # get perplexities from files
        pps = read_delta_file(file_path)

        # plot perplexities histogram and fit curve
        bin_edges = plot_histogram_fit(ax, pps, label, bin_nb, bin_edges)

    # compare pps from different files
    compare_pps(file_paths)
    # axs.set_title(title)


def get_unique_labels(fig):
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    print(lines_labels)
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    # grab unique labels
    unique_labels = list(set(labels))
    print('unique_labels', unique_labels)

    #override list
    unique_labels = ['sub', 'skewed fit sub', 'rel', 'skewed fit rel']

    # assign labels and legends in dict
    legend_dict = dict(zip(labels, lines))

    # query dict based on unique labels
    unique_lines = [legend_dict[x] for x in unique_labels]
    return unique_labels, unique_lines


def setup_plot(nbh_graphs):
    fig, axs = plt.subplots(1, nbh_graphs, figsize=(10, 2), dpi=300)

    # Beauty
    if nbh_graphs == 1:
        axs = [axs]
    for ax in axs:
        colors = iter([plt.cm.Set1(i) for i in [0, 0, 1, 1, 2, 2]])  # type: ignore
        ax.set_prop_cycle(cycler('color', colors))
        ax.spines['left'].set_visible(True)
        ax.axvline(0, color='black', linestyle='dotted')

    return fig, axs


if __name__ == "__main__":
    acc_file_dir = "output/synonyms/result/delta/"

    # models = ['allenai.OLMo-1B', 'allenai.OLMo-7B', 'llm360.Amber']
    # model = 'allenai.OLMo-1B'
    # controlled = ['delta2', 'delta3'] # random, controlled
    # ctrl = 'delta2'
    # test = 'T11'

    bins = 150

    # vary_cond(axs, test='T11', model='allenai.OLMo-1B', conditions=['delta2', 'delta3'], bin_nb=bins, labels=['random', 'controlled'])

    # vary_test(axs, tests=['T11', 'T12'], model='allenai.OLMo-1B', condition='delta3', bin_nb=bins, labels=['rel', 'sub'])

    # vary_model(axs, test='T11', models=['allenai.OLMo-1B', 'allenai.OLMo-7B', 'llm360.Amber'], condition='delta3', bin_nb=bins, labels=['OLMo-1B', 'OLMo-7B', 'Amber'])

    # vary(test='T11', model='allenai.OLMo-1B', condition=None, items=['delta2', 'delta3'], bin_nb=bins, labels=['random', 'controlled'], item_type='conditions')

    models = ['allenai.OLMo-1B', 'allenai.OLMo-7B', 'llm360.Amber']
    nb_horizontal_graphs = len(models)
    fig, axs = setup_plot(nb_horizontal_graphs)

    for i, model in enumerate(models):
        ax = axs[i]
        plot_from_files(ax, test_name='T11', model=model, condition='delta2', items=['T11', 'T12'], bin_nb=bins, labels=['rel', 'sub'], item_type='tests')

        if i == 0:
            ax.set_ylabel('Frequency', fontsize=14)
        if i == len(models) // 2:
            ax.set_xlabel('Delta pp', fontsize=14)
        ax.set_title(f'{model.split(".")[1]}', fontsize=14)
        #ax.set_xlim(-25, 10)

    unique_labels, unique_lines = get_unique_labels(fig)
    #plt.subplots_adjust(bottom=0.9)
    fig.legend(unique_lines, unique_labels,
               loc="outside lower center", borderaxespad=-0.4,
               ncol=4,
               frameon=False,
               fontsize=14)
    plt.subplots_adjust(bottom=0.8)
    fig.tight_layout()
    plt.show()
    # plot_from_files(test_name=None, model='allenai.OLMo-1B', condition='delta3', items=['T11', 'T12'], bin_nb=bins, labels=['rel', 'sub'], item_type='tests')

    # vary(test='T11', model=None, condition='delta3', items=['allenai.OLMo-1B', 'allenai.OLMo-7B', 'llm360.Amber'], bin_nb=bins, labels=['OLMo-1B', 'OLMo-7B', 'Amber'], item_type='models')
import os
from pathlib import Path
import numpy as np
from matplotlib.pyplot import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes


def get_data(out_file, pure_windows, opened=False, precision=True):
    result_data = np.loadtxt(open(out_file, "rb"), delimiter=",", skiprows=1)

    if pure_windows:
        start = 1
        start += 14 if opened else 0
        start += 0 if precision else 7
        end = start + 7
    else:
        start = 1
        end = 8

    return result_data[:, start:end]


# function for setting the colors of the box plots
def setBoxColors(bp, col):
    setp(bp['boxes'], color='blue' if col == 0 else 'red' if col == 1 else 'green')
    setp(bp['caps'], color='blue' if col == 0 else 'red' if col == 1 else 'green')
    setp(bp['whiskers'], color='blue' if col == 0 else 'red' if col == 1 else 'green')
    setp(bp['medians'], color='blue' if col == 0 else 'red' if col == 1 else 'green')
    setp(bp['fliers'], markeredgecolor='blue' if col == 0 else 'red' if col == 1 else 'green')


def get_boxplots(out_pure_win_file, out_all_win_file, boxplot_out_path, precision=True, show=False):
    # get the data
    opened_data = get_data(out_pure_win_file, pure_windows=True, opened=True, precision=precision)
    closed_data = get_data(out_pure_win_file, pure_windows=True, opened=False, precision=precision)
    all_data = get_data(out_all_win_file, pure_windows=False)

    figure()
    ax = axes()

    # Each column will be an algorithm -> a trio of boxplots
    for i in range(len(opened_data[1])):
        data = [opened_data[:, i]]
        initial_pos = 1 + i * 4
        bp = boxplot(data, positions=[initial_pos], widths=0.6)
        setBoxColors(bp, 0)
        data = closed_data[:, i]
        bp = boxplot(data, positions=[initial_pos + 1], widths=0.6)
        setBoxColors(bp, 1)
        data = all_data[:, i]
        bp = boxplot(data, positions=[initial_pos + 2], widths=0.6)
        setBoxColors(bp, 3)

    # set axes limits and labels
    final_pos = len(opened_data[1]) * 4
    xlim(0, final_pos)
    ylim(0.3, 1.2)
    ax.set_xticks([1.5 + i * 4 for i in range(len(opened_data[1]))])
    ax.set_xticklabels(['AB', 'DT', 'KNN', 'LDA', 'RF', 'QDA', 'SVM'])

    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1, 1], 'b-')
    hR, = plot([1, 1], 'r-')
    hG, = plot([1, 1], 'g-')
    legend((hB, hR, hG), ('Pure opened', 'Pure closed', 'All windows'))
    hB.set_visible(False)
    hR.set_visible(False)
    hG.set_visible(False)

    os.makedirs(boxplot_out_path, exist_ok=True)
    savefig(boxplot_out_path + '/boxplot.png')
    if show:
        show()


def get_paths(root_dir):
    routes_list = []
    for root, dirs, files in os.walk(root_dir):
        # Check if we are in the level of 10s, 5s, 8s folders
        if os.path.basename(root) in ['10s', '5s', '8s']:
            all_windows_csv = os.path.join(root, 'all_windows', 'full_results.csv')
            pure_windows_csv = os.path.join(root, 'pure_windows', 'full_results.csv')
            if os.path.exists(all_windows_csv) and os.path.exists(pure_windows_csv):
                out_boxplot_path = os.path.join(Path(root_dir).parent, 'boxplots', os.path.relpath(root, root_dir))
                routes_list.append((all_windows_csv, pure_windows_csv, out_boxplot_path))

    return routes_list


if __name__ == "__main__":
    routes = get_paths("../../../../out/results")
    for all_win_csv, pure_win_csv, out_boxplot in routes:
        get_boxplots(pure_win_csv, all_win_csv, out_boxplot)

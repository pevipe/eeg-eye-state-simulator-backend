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


# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['fliers'][0], color='red')
    setp(bp['fliers'][1], color='red')
    setp(bp['medians'][1], color='red')


def get_boxplots(out_file, pure_windows, precision=True):
    # get the data
    opened_data = get_data(out_file, pure_windows, opened=True, precision=precision)
    closed_data = get_data(out_file, pure_windows, opened=False, precision=precision)

    figure()
    ax = axes()

    # Each column will be an algorithm -> a pair of boxplots
    for i in range(len(opened_data[1])):
        data = [opened_data[:, i], closed_data[:, i]]
        initial_pos = 1 + i*3
        bp = boxplot(data, positions=[initial_pos, initial_pos + 1], widths=0.6)
        setBoxColors(bp)

    # set axes limits and labels
    final_pos = 1 + len(opened_data[1])*3
    xlim(0, final_pos)
    ylim(0.3, 1.2)
    ax.set_xticks([1.5 + i*3 for i in range(len(opened_data[1]))])
    ax.set_xticklabels(['AB', 'DT', 'KNN', 'LDA', 'RF', 'QDA', 'SVM'])

    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1, 1], 'b-')
    hR, = plot([1, 1], 'r-')
    legend((hB, hR), ('Closed', 'Opened'))
    hB.set_visible(False)
    hR.set_visible(False)

    savefig('boxcompare.png')
    show()


if __name__ == "__main__":
    out_filepath = "../../out/results/collective/10s/pure_windows/full_results.csv"
    get_boxplots(out_filepath, pure_windows=True, precision=True)

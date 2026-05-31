import numpy as np
from matplotlib import pyplot as plt
import warnings
from sklearn import metrics
from matplotlib import pylab
from prettytable import PrettyTable
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.patches import Polygon
from matplotlib.markers import MarkerStyle

warnings.filterwarnings('ignore')


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'TFMOA', 'TSA', 'BOA', 'POA', 'Proposed']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
    for j in range(len(Algorithm) - 1):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    length = np.arange(Fitness.shape[1])
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve')
    plt.plot(length, Fitness[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label=Algorithm[1])
    plt.plot(length, Fitness[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12, label=Algorithm[2])
    plt.plot(length, Fitness[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label=Algorithm[3])
    plt.plot(length, Fitness[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label=Algorithm[4])
    plt.plot(length, Fitness[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label=Algorithm[5])
    plt.xlabel('No. of Iteration', fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.ylabel('Cost Function', fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.legend(loc=1, prop={'weight': 'bold', 'size': 12})
    plt.savefig("./Results/Conv.png")
    plt.show()


def Plot_ROC_Curve():
    lw = 3
    cls = ['Ref 5', 'Ref 6', 'Ref 8', 'AHV-BRA ', 'Proposed']
    Actual = np.load('Target.npy', allow_pickle=True).astype(np.int32)
    lenper = round(Actual.shape[0] * 0.75)
    Actual = Actual[lenper:, :]
    fig = plt.figure()
    fig.canvas.manager.set_window_title('ROC Curve')
    colors = cycle(["blue", "#f4d35e", "limegreen", "deeppink", "black"])
    for i, color in zip(range(5), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True).astype(np.int32)[i]
        false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc * 100

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            lw=2,
            label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.ylabel("True Positive Rate", fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=15, fontweight='bold', color='#1d3557')
    plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='#1d3557')
    plt.title("ROC Curve")
    plt.legend(loc="lower right", prop={'weight': 'bold', 'size': 12})
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def Plot_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18]
    ActFun = ['Linear', 'ReLU', 'TanH', 'Softmax']
    Algorithm = ['TFMOA', 'TSA', 'BOA', 'POA', 'Proposed']
    Classifier = ['Ref 5', 'Ref 6', 'Ref 8', 'AHV-BRA ', 'Proposed']
    for j in range(len(Graph_Terms)):
        Graph = np.zeros(eval.shape[0:2])
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Terms[j] + 4]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_axes([0.16, 0.13, 0.7, 0.7])
        fig.canvas.manager.set_window_title(Terms[Graph_Terms[j]] + 'Algorithm Comparison of Learning Percentage')
        X = np.arange(len(ActFun))
        colors = ['#6a994e', '#00a8e8', 'violet', 'crimson', 'k']
        bars1 = plt.bar(X + 0.00, Graph[:, 0], color='#6a994e', width=0.15, label=Algorithm[0])
        bars2 = plt.bar(X + 0.15, Graph[:, 1], color='#00a8e8', width=0.15, label=Algorithm[1])
        bars3 = plt.bar(X + 0.30, Graph[:, 2], color='violet', width=0.15, label=Algorithm[2])
        bars4 = plt.bar(X + 0.45, Graph[:, 3], color='crimson', width=0.15, label=Algorithm[3])
        bars5 = plt.bar(X + 0.60, Graph[:, 4], color='k', width=0.15, label=Algorithm[4])

        for bars in [bars1, bars2, bars3, bars4, bars5]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height - (0.05 * height),
                         f"{str(np.round(height, 2))}", ha='center', va='top', fontsize=10, color='w', rotation=90,
                         fontweight='bold')

        dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                       in colors]
        plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.20), fontsize=9,
                   frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
        plt.xticks(X + 0.30, ['35', '45', '55', '65'], fontsize=15,
                   fontname="Arial",
                   fontweight='bold', color='k')
        plt.xlabel('Learning Percentage', fontname="Arial", fontsize=15, fontweight='bold', color='#14213d')
        plt.ylabel(Terms[Graph_Terms[j]], fontsize=15, fontname="Arial", fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='#35530a')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.tight_layout()
        path = "./Results/%s_Prop_Alg_Bar.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()

        # ------------------------------------- Methods ------------------------------------------------
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_axes([0.16, 0.13, 0.7, 0.7])
        fig.canvas.manager.set_window_title(Terms[Graph_Terms[j]] + 'Method Comparison of Learning Percentage')
        X = np.arange(len(ActFun))
        colors_1 = ['#f77f00', '#52796f', '#4361ee', '#ff0054', 'k']
        bars1 = plt.bar(X + 0.00, Graph[:, 5], color='#f77f00', width=0.15, label=Classifier[0])
        bars2 = plt.bar(X + 0.15, Graph[:, 6], color='#52796f', width=0.15, label=Classifier[1])
        bars3 = plt.bar(X + 0.30, Graph[:, 7], color='#4361ee', width=0.15, label=Classifier[2])
        bars4 = plt.bar(X + 0.45, Graph[:, 8], color='#ff0054', width=0.15, label=Classifier[3])
        bars5 = plt.bar(X + 0.60, Graph[:, 4], color='k', width=0.15, label=Classifier[4])

        for bars in [bars1, bars2, bars3, bars4, bars5]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height - (0.05 * height),
                         f"{str(np.round(height, 2))}", ha='center', va='top', fontsize=10, color='w', rotation=90,
                         fontweight='bold')

        dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                       in colors_1]
        plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.20), fontsize=9,
                   frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
        plt.xticks(X + 0.30, ['35', '45', '55', '65'], fontsize=15,
                   fontname="Arial",
                   fontweight='bold', color='k')
        plt.xlabel('Learning Percentage', fontname="Arial", fontsize=15, fontweight='bold', color='#14213d')
        plt.ylabel(Terms[Graph_Terms[j]], fontsize=15, fontname="Arial", fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='#35530a')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.tight_layout()
        path = "./Results/%s_Prop_Mod_Bar.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()


def Table():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Algorithm = ['Step Per Epochs', 'TFMOA', 'TSA', 'BOA', 'POA', 'Proposed']
    Classifier = ['Step Per Epochs', 'Ref 5', 'Ref 6', 'Ref 8', 'AHV-BRA ', 'Proposed']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 2, 4, 8, 9, 12, 15, 17, 18]).astype(int)
    Table_Terms = [0, 2, 4, 8, 9, 12, 15, 17, 18]
    table_terms = [Terms[i] for i in Table_Terms]
    StepPerEpochs = [100, 200, 300, 400, 500]
    for k in range(len(Table_Terms)):
        value = eval[:, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], StepPerEpochs)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
        print('----------------------------------------- ', table_terms[k], '  Algorithm Comparison',
              '---------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], StepPerEpochs)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
        print('------------------------', table_terms[k], '  Classifier Comparison',
              '---------------------------------------')
        print(Table)


def Detection_Plots_Results():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Precision', 'F1 Score', 'Sensitivity', 'MAP']
    Graph_Terms = [0, 1, 2, 3, 4]
    bar_width = 0.15
    Image = ['256', '512', '720', '1000']
    Classifier = ['Unet', 'Unet3+', 'ResUnet', 'TransUnet', 'Proposed']
    for j in range(len(Graph_Terms)):
        Graph = np.zeros(eval.shape[0:2])
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Terms[j] + 4]

        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
        fig.canvas.manager.set_window_title('Method Comparison of Image Resolution')
        X = np.arange(len(Image))
        plt.bar(X + 0.00, Graph[:, 0], color='darkblue', edgecolor='w', linewidth=2, width=0.15,
                label=Classifier[0])
        plt.bar(X + 0.15, Graph[:, 1], color='#9400d3', edgecolor='w', linewidth=2, width=0.15,
                label=Classifier[1])
        plt.bar(X + 0.30, Graph[:, 2], color='#a30046', edgecolor='w', linewidth=2, width=0.15,
                label=Classifier[2])
        plt.bar(X + 0.45, Graph[:, 3], color='#00bbf9', edgecolor='w', linewidth=2, width=0.15,
                label=Classifier[3])
        plt.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', linewidth=2, width=0.15,
                label=Classifier[4])
        plt.xticks(X + bar_width * 2, ['256p', '512p', '720p', '1000p'], fontsize=14,
                   fontname="Arial",
                   fontweight='bold', color='k')
        plt.xlabel('Image Resolution', fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')
        plt.ylabel(Terms[Graph_Terms[j]], fontsize=14, fontname="Arial", fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
        # Remove axes outline
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                       in ['darkblue', '#9400d3', '#a30046', '#00bbf9', 'k']]
        plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=10,
                   frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
        plt.tight_layout()
        path = "./Results/%s_bar_Face_Detection.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()


if __name__ == '__main__':
    plotConvResults()
    Plot_Results()
    Plot_ROC_Curve()
    Table()
    Detection_Plots_Results()

import numpy as np
import pandas as pd
import random
import util
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from statsmodels.nonparametric.kernel_density import  KDEMultivariate
from KDEpy import FFTKDE

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kdt = KDTree(X, leaf_size=50, metric='euclidean')
kdt.query(X, k=2, return_distance=False)
eps = 0.01


def visualize_histogram_2d(data):
    hist, xedges, yedges = np.histogram2d(data['x'], data['y'], bins=(11, 11))
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, hist, cmap=plt.get_cmap('Greys'))
    plt.show()


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def histogram_of_class(data, label):
    subset = data.loc[data['label'] == label]
    hist, xedges, yedges = np.histogram2d(subset['x'], subset['y'], bins= (6,6))
    hist = hist + eps
    return hist / np.sum(hist)





def get_pointset_name(n):
    return 'pointsets/pointset{}.csv'.format(n)


def moran(d, w):
    d = d[:, -1]
    N = d.shape[0]
    W = np.sum(w)
    mean = np.mean(d)
    summation = 0
    for i in range(N):
        for j in range(N):
            if w[i, j] != 0:
                summation = summation + (w[i, j] * (d[i] - mean) * (d[j] - mean))

    denum = 0
    for i in range(N):
        v = d[i] - mean
        denum += v**2
    return (N/W)*(summation/denum)


def euclidean_dist(x, y):
    return np.linalg.norm(np.array(x)-np.array(y))


def get_weight_matrix(data):
    N = data.shape[0]
    w = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dist = euclidean_dist(data[i, :-1], data[j, :-1])
                if dist < 2:
                    if dist < 1:
                        w[i, j] = 1
                    else:
                        w[i, j] = 1/ np.power(dist, 2)
    return w


def get_weight_matrix_knn(data, k=10):
    data = data[:, :-1]
    n = data.shape[0]
    w = np.zeros((n,n))
    d = np.array(data)

    for i in range(n):
        kdt = KDTree(d, leaf_size=20, metric='euclidean')
        knn = kdt.query(d, k=k, return_distance=False)
        for j in range(1, len(knn[i])):
            w[i, knn[i][j]] = 1
    return w


def randomize_data(data, percentage):
    d = np.array(data)
    n = int(np.round(d.shape[0] * percentage))
    for i in range(n):
        indx = random.randrange(0, d.shape[0])
        val = d[indx, -1]
        d[indx, -1] = (val + 1) % 2
    return d


def alter_labels(data, list):
    for i in list:
        data[i, -1] = (1+ data[i, -1])% 2

    return data


def plot_various_randomness(data, output_name, increment=0.05):
    counter = 1
    size = data.shape[0]
    # w = get_weight_matrix(data)
    w = get_weight_matrix_knn(data)
    n = int(np.round(size * 0.05))
    d = np.array(data)
    random_list = random.sample(range(size), 6*n)
    fig, axs = plt.subplots(2, 3)
    for i in range(2):
        for j in range(3):
            d = alter_labels(d, random_list[(counter - 1) * n : counter * n])
            color = ['red' if l == 0 else 'blue' for l in d[:, -1]]
            d0 = d[d[:, -1] == 0, :]
            d1 = d[d[:, -1] == 1, :]
            p = get_kernel_distribution_estimation(d0, 32)
            q = get_kernel_distribution_estimation(d1, 32)
            m = kl_divergence(p, q) + kl_divergence(q, p)
            # m = moran(d, w)
            axs[i, j].scatter(data['x'], data['y'], color=color, marker='.')
            axs[i, j].set_title('measure = {:.3f}'.format(m))
            counter += 1

    for ax in axs.flat:
        ax.label_outer()
        fig.supplot("KL_Divergence method", y = 0.8)
    plt.savefig(output_name)
    plt.show()


def get_kernel_distribution_estimation(data, grid, kernel='gaussian', norm=2):
    kde = FFTKDE(kernel=kernel, norm=norm)
    grid, pdf = kde.fit(data).evaluate(grid)
    return pdf/np.sum(pdf)


def kl_divergence_kernel_dist_estimation(data, grid, kernel="gaussian", norm=2):
    d0 = data[data[:, -1] == 0, :]
    d1 = data[data[:, -1] == 1, :]
    p = get_kernel_distribution_estimation(d0, grid, kernel, norm)
    q = get_kernel_distribution_estimation(d1, grid, kernel, norm)
    measure = kl_divergence(p,q) + kl_divergence(q,p)
    return measure


def average_moransi_randomized_data(data, random_percent, repeat=50):
    measures = []
    for i in range(repeat):
        d_r = randomize_data(data, random_percent)
        measures.append(moran(d_r, get_weight_matrix_knn(data)))
    return measures


def average_kl_div_randomized_data(data, random_percent, repeat=50):
    measures = []
    for i in range(repeat):
        if (i+1) % 5 ==0:
            print ('roound {}'.format(i+1))
        d_r = randomize_data(data, random_percent)
        measures.append(kl_divergence_kernel_dist_estimation(d_r, grid=128))
    return measures

files = ['inner_circles.csv',  'pointset8.csv', 'cross.csv', 'S_curve.csv', 'crossing_recs.csv', 'moons.csv' , 'random.csv']
data = np.array(pd.read_csv('pointsets/{}'.format(files[4])))
measures_moran = np.zeros((50, 7))
measures_list = []
for i in range(1, 8):
    print("{}% randomized in process".format(i*10))
    temp = average_moransi_randomized_data(data, 0.1*i, 50)
    measures_list.append(temp)
    measures_moran[:, i-1] = temp
overview_morans.append(measures_moran)


measures = []
name = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
fig, axs = plt.subplots(2,4)
counter = 0
for i in range(2):
    for j in range(4):
        print(files[counter])
        dir = 'pointsets/{}'.format(files[counter])
        data = pd.read_csv(dir)
        data = np.array(data)

        measure = moran(data, get_weight_matrix_knn(data, k=10))
        measures.append(measure)
        print(measure)
        # print(measure)
        # measure = kl_divergence_kernel_dist_estimation(data, grid=128)
        # measure = get_measure_kldiv(data)

        color = ['blue' if l == 0 else 'red' for l in data[:, -1]]
        if counter == 0:
            axs[i, j].set_xlim((-5, 5))
            axs[i, j].set_ylim((-6, 6))
        if counter == 2:
            axs[i, j].set_ylim((-10, 13))
        if counter == 3:
            axs[i, j].set_ylim((-2, 18))
        if counter == 6:
            axs[i, j].set_ylim((-2, 2))
        axs[i, j].scatter(data[:, 0], data[:, 1], color=color, marker='.', s=1)
        axs[i, j].set_title('({.3f})'.format(measure))
        counter += 1
        axs[i, j].tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off

for ax in axs.flat:
    ax.label_outer()

# plt.subplots_adjust(top=0.8)
plt.savefig('moransi.pdf', format='pdf', dpi=800)
plt.show()

measures_moran = []
counter = 0
for i in range(len(files)):
    print(files[i])
    dir = 'pointsets/{checker}'.format(files[i])
    data = pd.read_csv(dir)
    data = np.array(data)
    l_k = []
    for k in range(5, 55, 5):
        measure = moran(data, get_weight_matrix_knn(data, k=k))
        l_k.append(measure)
    counter += 1
    measures_moran.append(l_k)
    print(l_k)


checker_moran = []
dir = 'pointsets/checkerboard.csv'
data = pd.read_csv(dir)
data = np.array(data)
for k in range(5, 55, 5):
    print("k = {}".format(k))
    measure = moran(data, get_weight_matrix_knn(data, k=k))
    checker_moran.append(measure)


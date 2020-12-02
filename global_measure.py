import numpy as np
import random
import util
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from KDEpy import FFTKDE
import KDEpy

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
    sum = 0
    for i in range(p.shape[0]):
        if p[i] >= 1e-9 :
            sum += p[i] * np.log(p[i] / q[i])
    return sum
    # return np.sum(np.where(p != 0, p * np.log(p / q), 0))


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
    n = data.shape[0]
    w = np.zeros((n,n))

    kdt = KDTree(data[:, :-1], leaf_size=20, metric='euclidean')
    knn = kdt.query(data[:, :-1], k=k, return_distance=False)
    for i in range(n):
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
            p = get_kernel_density_estimation(d0, 32)
            q = get_kernel_density_estimation(d1, 32)
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


def get_kernel_density_estimation(data, grid, bw=1, kernel='gaussian', norm=2):
    kde = FFTKDE(kernel=kernel, norm=norm, bw=bw)
    grid, pdf = kde.fit(data).evaluate(grid)
    return pdf/np.sum(pdf)


def get_KDE_visualization(data,  N=16, bw=1, grid_points = 2**7, kernel='gaussian', norm=2):
    # Compute the kernel density estimate
    xmin, xmax = np.min(data[:, 0]), np.max(data[:, 0])
    ymin, ymax = np.min(data[:, 1]), np.max(data[:, 1])
    grid_x = np.linspace(xmin - (xmax - xmin) * 0.2, xmax + (xmax - xmin) * 0.2, grid_points)
    grid_y = np.linspace(ymin - (ymax - ymin) * 0.2, ymax + (ymax - ymin) * 0.2, grid_points)
    grid = np.stack(np.meshgrid(grid_y, grid_x), -1).reshape(-1, 2)
    grid[:, [0, 1]] = grid[:, [1, 0]]  # Swap indices
    pdf = KDEpy.FFTKDE(kernel=kernel, norm=norm, bw=bw).fit(data[:, :-1]).evaluate(grid)
    # The grid is of shape (obs, dims), points are of shape (obs, 1)
    x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    z = pdf.reshape(grid_points, grid_points).T

    # Plot the kernel density estimate
    # plt.contour(x, y, z, N, linewidths=0.4, colors='k')
    # plt.contourf(x, y, z, N, cmap="RdBu_r")
    plt.pcolormesh(x, y, z, cmap=plt.get_cmap('RdBu_r'))
    plt.colorbar()
    # plt.plot(data[:, 0], data[:, 1], 'ok', ms=3)
    plt.show()


def kl_divergence_kernel_dist_estimation(data, grid, bw=1, kernel="gaussian", norm=2):
    d0 = data[data[:, -1] == 0, :-1]
    d1 = data[data[:, -1] == 1, :-1]
    p = get_kernel_density_estimation(d0, grid, bw, kernel, norm)
    q = get_kernel_density_estimation(d1, grid, bw, kernel, norm)
    measure = kl_divergence(p,q) + kl_divergence(q,p)
    return measure


def kl_divergence_kernel_dist_estimation_mesh(data, grid_points=128, bw=1, kernel="gaussian", norm=2):
    d0 = data[data[:, -1] == 0, :-1]
    d1 = data[data[:, -1] == 1, :-1]
    xmin , xmax = np.min(data[:, 0]), np.max(data[:,0])
    ymin, ymax = np.min(data[:, 1]), np.max(data[:,1])
    grid_x = np.linspace(xmin - (xmax-xmin)*0.2, xmax + (xmax-xmin)*0.2, grid_points)
    grid_y = np.linspace(ymin - (ymax-ymin)*0.2, ymax + (ymax-ymin)*0.2, grid_points)
    yv, xv = np.meshgrid(grid_y, grid_x)
    mesh_grid = np.hstack((xv.reshape((-1,1)), yv.reshape((-1,1))))
    # p = KDEpy.FFTKDE(kernel=kernel, norm=norm, bw=bw).fit(d0).evaluate(mesh_grid)
    # q = KDEpy.FFTKDE(kernel=kernel, norm=norm, bw=bw).fit(d1).evaluate(mesh_grid)
    p = KDEpy.TreeKDE(kernel=kernel, bw=bw, norm=norm).fit(d0).evaluate(mesh_grid)
    q = KDEpy.TreeKDE(kernel=kernel, norm=norm, bw=bw).fit(d1).evaluate(mesh_grid)
    p = p/(np.sum(p))
    q = q/(np.sum(q))
    measure = kl_divergence(p, q) + kl_divergence(q, p)
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


def visualise_measures(dir):
    measures = np.load(dir)


def calculate_kl_div_measure_all_files(folder, main_dir, trial_range, n_steps, grid_points=128, bw=1):
    arr = np.ones((trial_range[1], n_steps))
    # if trial_range[0] != 0:
    #     old = np.load('measures/kl_div/{}.npy'.format(folder[:-1]))
    #     arr[:old.shape[0], :] = old

    path = main_dir+folder
    files = util.get_file_names(main_dir+folder)
    for f in files:
        if f[-4:] == '.npy':
            name = f[:-4]
            step, trial = name.split('_')[-2:]
            step, trial = int(int(step)/10), int(trial)
            if trial < trial_range[0] or trial >= trial_range[1]:
                continue
            data = np.load(path+f)
            print(path+f)
            # print('trial={}, step={}'.format(trial, step))
            m = kl_divergence_kernel_dist_estimation_mesh(data, grid_points=grid_points, bw=bw)
            arr[trial, step] = m
    return arr


def calculate_moransi_measure_all_files(folder, main_dir, trial_range, n_steps, k=15):
    arr = np.ones((trial_range[1], n_steps))
    if trial_range[0] != 0:
        old = np.load('measures/kl_div/{}.npy'.format(folder[:-1]))
        arr[:old.shape[0], :] = old
    path = main_dir+folder
    files = util.get_file_names(main_dir+folder)
    for f in files:
        if f[-4:] == '.npy':
            name = f[:-4]
            step, trial = name.split('_')[-2:]
            step, trial = int(int(step)/10), int(trial)
            data = np.load(path+f)
            print(path + f)
            # print('trial={}, step={}'.format(trial, step))
            m = moran(data, get_weight_matrix_knn(data, k=k))
            arr[trial, step] = m
    return arr


dict_bw = {
            'circles_gm': 1,
            'circles_pm' : 1,
            'inner_circle_pm' : 1,
            'checkerboard_pm' : 0.05,
            'cross_pm' : 1,
            'crossing_recs_pm': 0.5,
            'moons_pm' : 0.5,
            's_curve_pm' : 0.5,
            'minkowski_s_curve' : 0.5,
            'minkowski_moons' : 0.5,
            'minkowski_cross' : 1,
            'minkowski_inner_circle' : 1,
            'minkowski_crossing_recs' : 0.5,
            'minkowski_checkerboard' : 0.5
}

folders = [
            # 'circles_gm',
            # 'circles_pm',
            # 'inner_circle_pm',
            # 'checkerboard_pm',
            # 'cross_pm',
            # 'crossing_recs_pm',
            'moons_pm',
            # 's_curve_pm',
# 'minkowski_s_curve',
'minkowski_moons',
# 'minkowski_cross',
# 'minkowski_inner_circle',
# 'minkowski_crossing_recs',
# 'minkowski_checkerboard'
]

g =32
bw = 0.5

for f in folders:
    if f[-2:] == 'pm':
        arr_pm = calculate_kl_div_measure_all_files(f + '/', 'pointsets/', (0,10), 8, grid_points=g, bw=bw)
        np.save('measures/kl_div/{}.npy'.format(f), arr_pm)
    else:
        arr_gm = calculate_kl_div_measure_all_files(f + '/', 'pointsets/', (0,10), 11, grid_points=g, bw=bw)
        if f[-2:] == 'gm':
            np.save('measures/kl_div/{}.npy'.format(f), arr_gm)
        else :
            np.save('measures/kl_div/{}.npy'.format(f[10:] + "_gm"), arr_gm)

for f in folders:
    measures = np.load('measures/kl_div/{}.npy'.format(f))
    # plt.plot(np.mean(measures, axis=0))
    plt.boxplot(measures)
    plt.title('kl_divergence {}'.format(f))
    plt.savefig('measures/kl_div_{}.png'.format(f))
    plt.show()

for f in folders:
    print(f)
    if f[-2:] == 'pm':
        arr = calculate_moransi_measure_all_files(f + '/', 'pointsets/', (0,11), 8)
        np.save('measures/moransi/{}.npy'.format(f), arr)
    else:
        arr = calculate_moransi_measure_all_files(f + '/', 'pointsets/', (0,11), 11)
        np.save('measures/moransi/{}.npy'.format(f), arr)


for f in folders:
    measures = np.load('measures/moransi/{}.npy'.format(f))
    # plt.plot(np.mean(measures, axis=0))
    plt.boxplot(measures)
    plt.title('kl_divergence {}'.format(f))
    plt.savefig('measures/moransi_{}.png'.format(f))
    plt.show()

m = []
for i in range(11):
    data = np.load('pointsets/circles_gm/circles_gm_0_{}.npy'.format(i))
    util.plot_points_2d_np(data, plt, 'out.png')
    # data = np.load('pointsets/circles_pm/circles_pm_0_0.npy')
    # util.plot_points_2d_np(data, plt, 'out.png')

    # Create 2D grid
    grid_x = np.linspace(0, 1, 2 ** 7)
    grid_y = np.linspace(0, 1, 2 ** 7)
    grid = np.stack(np.meshgrid(grid_y, grid_x), -1).reshape(-1, 2)
    grid[:, [0, 1]] = grid[:, [1, 0]] # Swap indices

    # density estimates

    y1 = KDEpy.FFTKDE(bw=1).fit(data[data[:, -1]==1, :-1]).evaluate(grid)
    y2 = KDEpy.FFTKDE(bw=1).fit(data[data[:, -1]==0, :-1]).evaluate(grid)

    y1 = y1/np.sum(y1)
    y2 = y2/np.sum(y2)
    m .append(kl_divergence(y1, y2) + kl_divergence(y2, y1))

y2 = KDEpy.TreeKDE(bw=0.2).fit(data[:, :-1]).evaluate(grid)
y3 = KDEpy.NaiveKDE(bw=0.2).fit(data[:, :-1]).evaluate(grid)
# plots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].set_title("Data")
axes[0, 0].scatter(data[:, 0], data[:, 1], marker="x", color="red")
axes[0, 1].set_title("FFTKDE")
axes[0, 1].scatter(grid[:, 0], grid[:, 1], c=y1)
# axes[0, 1].scatter(data[:, 0], data[:, 1], marker="x", color="red")
axes[1, 0].set_title("TreeKDE")
axes[1, 0].scatter(grid[:, 0], grid[:, 1], c=y2)
# axes[1, 0].scatter(data[:, 0], data[:, 1], marker="x", color="red")
axes[1, 1].set_title("Naive")
axes[1, 1].scatter(grid[:, 0], grid[:, 1], c=y3)
# axes[1, 1].scatter(data[:, 0], data[:, 1], marker="x", color="red")
plt.show()


plt.boxplot(m)
plt.show()
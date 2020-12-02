import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.neighbors import KDTree
import util
import matplotlib.pyplot as plt


def make_labels(n_reds, n_blues):
    return np.repeat(np.array([1, 0]), [n_reds, n_blues], axis=0)


def randomize_data(data, percentage):
    n = int(np.round(data.shape[0] * percentage/100))
    indx = np.random.randint(0, data.shape[0], n)
    for i in indx:
        val = data[i, -1]
        data[i, -1] = (val + 1) % 2
    return data


def make_hyper_sphere(dimension, n_samples, r=1):
    sphere = np.empty((0, dimension))

    for i in range(n_samples):
        p = np.random.uniform(-r, r, dimension)
        while np.linalg.norm(p) > r:
            p = np.random.uniform(-r, r, dimension)
        sphere = np.vstack((sphere, p))
    return sphere


def make_hyper_cube(dimension, n_samples, r=1):
    cube = np.empty((0, dimension))
    for i in range(n_samples):
        p = np.random.uniform(-r, r, dimension)
        cube = np.vstack((cube, p))
    return cube


def make_hyper_hollow_sphere(dimension, n_samples, outer_r, inner_r):
    donut = np.empty((0, dimension))
    for i in range(n_samples):
        p = np.random.uniform(-outer_r, outer_r, dimension)
        d = np.linalg.norm(p)
        while d > outer_r or d < inner_r:
            p = np.random.uniform(-outer_r, outer_r, dimension)
            d = np.linalg.norm(p)
        donut = np.vstack((donut, p))
    return donut


def calculate_kl_div_measure(data, bw=1):
    d0 = data[data[:, -1] == 0, :-1]
    d1 = data[data[:, -1] == 1, :-1]
    vt = 'c' * d0.shape[1]

    dens_d0 = sm.nonparametric.KDEMultivariate(d0, var_type=vt, bw=[bw] * d0.shape[1])
    dens_d1 = sm.nonparametric.KDEMultivariate(d1, var_type=vt, bw=[bw] * d0.shape[1])

    pdf_0 = dens_d0.pdf(data[:, :-1])
    pdf_1 = dens_d1.pdf(data[:, :-1])
    pdf_0 = pdf_0 / np.sum(pdf_0)
    pdf_1 = pdf_1 / np.sum(pdf_1)
    return kl_divergence(pdf_0, pdf_1) + kl_divergence(pdf_1, pdf_0)


def calculate_kl_div_best_bw(data, bw='cv_ls'):
    d0 = data[data[:, -1] == 0, :-1]
    d1 = data[data[:, -1] == 1, :-1]
    vt = 'c' * d0.shape[1]

    dens_d0 = sm.nonparametric.KDEMultivariate(d0, var_type=vt, bw=bw)
    dens_d1 = sm.nonparametric.KDEMultivariate(d1, var_type=vt, bw=bw)

    pdf_0 = dens_d0.pdf(data[:, :-1])
    pdf_1 = dens_d1.pdf(data[:, :-1])
    pdf_0 = pdf_0 / np.sum(pdf_0)
    pdf_1 = pdf_1 / np.sum(pdf_1)
    return kl_divergence(pdf_0, pdf_1) + kl_divergence(pdf_1, pdf_0)


def geometric_mixing_spheres(r1, r2, n_dim, distance, n_samples, n_steps, trial_range):
    c2_x = r1+r2+distance
    step = c2_x/(n_steps * 2)
    p = int(np.ceil(100/n_steps))
    for i in range(n_steps+1):
        for j in range(*trial_range):
            c1 = np.repeat(np.array([i * step, 0]), [1, n_dim-1])
            c2 = np.repeat(np.array([c2_x - (i*step), 0]), [1, n_dim - 1])
            s1 = make_hyper_sphere(n_dim, n_samples, r1) + c1
            s2 = make_hyper_sphere(n_dim, n_samples, r2) + c2
            data = np.vstack((s1, s2))
            labels = make_labels(n_samples, n_samples)
            data = np.hstack((data, labels.reshape((-1, 1))))
            np.save('pointsets/high_dimensional/{}D/hyper_spheres_d{}_{}_{}.npy'.format(n_dim, n_dim, i*p, j), data)


def geometric_mixing_hyper_inner_balls(inner_r, outer_r, n_dim, n_samples, n_steps, trial_range):
    step = (outer_r - inner_r) / (n_steps)
    p = int(np.ceil(100 / n_steps))
    for i in range(n_steps + 1):
        for j in range(*trial_range):
            s = make_hyper_sphere(n_dim, n_samples, inner_r)
            hs = make_hyper_hollow_sphere(n_dim, n_samples, outer_r, inner_r)

            noise_s = make_hyper_sphere(n_dim, n_samples, step*i)
            noise_hs = make_hyper_sphere(n_dim, n_samples, step*i)
            data = np.vstack((s + noise_s, hs + noise_hs))

            labels = make_labels(n_samples, n_samples)
            data = np.hstack((data, labels.reshape((-1, 1))))
            output_name = 'pointsets/high_dimensional/{}D/hyper_inner_balls_gm/hyper_inner_balls_d{}_{}_{}.npy'
            np.save(output_name.format(n_dim, n_dim, i*p, j), data)


def probabilistic_mixing_hyper_inner_balls(inner_r, outer_r, n_dim, n_samples, n_steps, trial_range):
    p = int(np.ceil(70 / n_steps))
    for i in range(n_steps + 1):
        for j in range(*trial_range):
            s = make_hyper_sphere(n_dim, n_samples, inner_r)
            hs = make_hyper_hollow_sphere(n_dim, n_samples, outer_r, inner_r)

            data = np.vstack((s, hs))
            labels = make_labels(n_samples, n_samples)
            data = np.hstack((data, labels.reshape((-1, 1))))

            data = randomize_data(data, i*p)
            output_name = 'pointsets/high_dimensional/{}D/hyper_inner_balls_pm/hyper_inner_balls_d{}_{}_{}.npy'
            np.save(output_name.format(n_dim, n_dim, i*p, j), data)


def probabilistic_mixing_hyper_spheres(r1, r2, n_dim, distance, n_samples, n_steps, trial_range):
    p = int(np.ceil(70 / n_steps))
    c2_x = r1 + r2 + distance
    for i in range(n_steps + 1):
        for j in range(*trial_range):
            c2 = np.repeat(np.array([c2_x, 0]), [1, n_dim - 1])
            s1 = make_hyper_sphere(n_dim, n_samples, r1)
            s2 = make_hyper_sphere(n_dim, n_samples, r2) + c2

            data = np.vstack((s1, s2))
            labels = make_labels(n_samples, n_samples)
            data = np.hstack((data, labels.reshape((-1, 1))))

            data = randomize_data(data, i*p)
            output_name = 'pointsets/high_dimensional/{}D/hyper_spheres_pm/hyper_spheres_d{}_{}_{}.npy'
            np.save(output_name.format(n_dim, n_dim, i*p, j), data)



def kl_divergence(p, q):
    sum = 0
    for i in range(p.shape[0]):
        if p[i] >= 1e-9  and q[i] >= 1e-9:
            sum += p[i] * np.log(p[i] / q[i])
    return sum


def moran(data, k=15):
    w = get_weight_matrix_knn(data, k=k)
    d = data[:, -1]
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
    return (N/W)*np.abs(summation/denum)


def get_weight_matrix_knn(data, k=15):
    n = data.shape[0]
    w = np.zeros((n,n))

    kdt = KDTree(data[:, :-1], leaf_size=20, metric='euclidean')
    knn = kdt.query(data[:, :-1], k=k, return_distance=False)
    for i in range(n):
        for j in range(1, len(knn[i])):
            w[i, knn[i][j]] = 1
    return w


def get_kl_div_measures(D, directory, n_trials=10, bw=1):
    name = directory[:-3]
    type ='geometric_mixing'
    n_steps = 5
    p = 20
    if directory[-2:] == 'pm':
        type = 'probabilistic_mixing'
        p = 10
        n_steps = 7
    measures = np.zeros((n_trials, n_steps+1))
    for i in range(n_steps+1):
        for j in range(n_trials):
            print('{}_d{}_{}_{}.npy'.format(name, D, i * p, j))
            path = directory + '/' + name
            data = np.load('pointsets/high_dimensional/{}D/{}_d{}_{}_{}.npy'.format(D, path, D, i * p, j))
            if isinstance(bw, str):
                measures[j, i] = calculate_kl_div_best_bw(data, bw=bw)
            else:
                measures[j, i] = calculate_kl_div_measure(data, bw=bw)

    np.save('measures/high_dimensional/kl_div/{}/kl_div_{}_{}D.npy'.format(type, name, D), measures)
    return measures


def get_moransi_measures(D, directory, n_trials=10, k=15):
    name = directory[:-3]
    type = 'geometric_mixing'
    n_steps = 5
    p = 20
    if directory[-2:] == 'pm':
        type = 'probabilistic_mixing'
        p = 10
        n_steps = 7
    measures = np.zeros((n_trials, n_steps + 1))
    # old_m = np.load('measures/high_dimensional/moransi/{}/moransi_{}_{}D.npy'.format(type, name, D))
    # if old_m.shape[1] == measures.shape[1]:
    #     return old_m
    # measures[:, :-2] = old_m
    for i in range(n_steps+1):
        # if i < 6:
        #     continue
        for j in range(n_trials):
            print('{}_d{}_{}_{}.npy'.format(name, D, i * p, j))
            path = directory + '/' + name
            data = np.load('pointsets/high_dimensional/{}D/{}_d{}_{}_{}.npy'.format(D, path, D, i * p, j))
            measures[j, i] = moran(data, k=k)
    np.save('measures/high_dimensional/moransi/{}/moransi_{}_{}D.npy'.format(type, name, D), measures)
    return measures


if __name__=='__main__':


    measures = np.zeros((10, 6))
    for i in range(6):
        for j in range(10):
            print('hyper_spheres_d{}_{}_{}.npy'.format(D, i * 20, j))
            measures[j, i] = calculate_kl_div_best_bw(
                np.load('pointsets/high_dimensional/{}D/hyper_spheres_d{}_{}_{}.npy'.format(D, D, i * 20, j)))
    np.save('measures/high_dimensional/kl_div/kl_div_hyper_spheres_{}D.npy'.format(D), measures)


    D = 4
    bw = 0.5
    directory = 'hyper_' + 'inner_balls' +'_gm'
    name = directory[:-3]
    p = 10
    n_steps = 7
    n_trials=10
    measures = np.zeros((n_trials, n_steps + 1))
    # measures[:, :6] = np.load("measures\high_dimensional\kl_div\probabilistic_mixing\kl_div_{}_{}D.npy".format(name, D))
    for i in range(n_steps + 1):
        for j in range(n_trials):
            print('{}_d{}_{}_{}.npy'.format(name, D, i * p, j))
            path = directory + '/' + name
            data = np.load('pointsets/high_dimensional/{}D/{}_d{}_{}_{}.npy'.format(D, path, D, i * p, j))
            if isinstance(bw, str):
                measures[j, i] = calculate_kl_div_best_bw(data, bw=bw)
            else:
                measures[j, i] = calculate_kl_div_measure(data, bw=bw)
    np.save('measures/high_dimensional/kl_div/probabilistic_mixing/kl_div_{}_{}D.npy'.format( name, D), measures)

directory = 'hyper_' + 'inner_balls' +'_pm'
for i in range(3,11):
    get_moransi_measures(i, directory, 10)
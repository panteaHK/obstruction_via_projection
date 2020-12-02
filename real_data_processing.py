from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import util
import NN_Learner as NN
import high_dimensional_data as hd
from sklearn.neighbors import KDTree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import time, math


def randomly_project_data(data):
    projected = np.zeros((data.shape[0], data.shape[1]-1))
    v = np.random.uniform(-1, 1, data.shape[1]-1)

    projected[:, :-1] = util.project_points(data[:, :-1], v)
    labels = data[:, -1]
    projected[:, -1] = labels
    return projected, v


def moran(d, k=15):
    weight_matrix = get_weight_matrix_knn(d, k=k)
    d = d[:, -1]
    N = d.shape[0]
    W = np.sum(weight_matrix)
    mean = np.mean(d)
    summation = 0
    for i in range(N):
        for j in range(N):
            if weight_matrix[i, j] != 0:
                summation = summation + (weight_matrix[i, j] * (d[i] - mean) * (d[j] - mean))
    denum = 0
    for i in range(N):
        v = d[i] - mean
        denum += v**2
    return (N/W)*(summation/denum)


def get_weight_matrix_knn(data, k=10):
    n = data.shape[0]
    w = np.zeros((n,n))
    kdt = KDTree(data[:, :-1], leaf_size=20, metric='euclidean')
    knn = kdt.query(data[:, :-1], k=k, return_distance=False)
    for i in range(n):
        for j in range(1, len(knn[i])):
            w[i, knn[i][j]] = 1
    return w


def get_all_measures(data, bw):
    # accs = nn.run_real_model_test_train_split(data, 0.3, 10, 20)
    accs = NN.run_model_test_train_split(data, 0.3, 10, 20)
    kl_div = hd.calculate_kl_div_measure(data, bw=bw)
    # kl_div = hd.calculate_kl_div_best_bw(data)
    moransi = moran(data, k=15)
    return (np.mean(accs), np.std(accs), kl_div, moransi)


def get_svm_separator(data, cv=5):
    X, Y = data[:, :-1], data[:, -1]
    param_grid = {'C': C}
    grid = GridSearchCV(LinearSVC(), param_grid, verbose=0, cv=cv)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print(grid.best_params_)
    print(clf.coef_/(np.linalg.norm(clf.coef_)))
    print(accuracy_score(Y, clf.predict(X)))
    vec = clf.coef_
    return vec


def make_small_adult(adult):
    adult_0 = adult[adult[:, -1] == 0]
    adult_1 = adult[adult[:, -1] == 1]
    adult_small = adult_0[np.random.randint(0, adult_0.shape[0], 5000)]
    adult_small = np.vstack((adult_small, adult_1[np.random.randint(0, adult_1.shape[0], 5000)]))


def find_epsilon_close_vectors(vec, epsilon=5, n=1000):
    radian_e = math.radians(epsilon)
    e_neighbors = np.empty((0, vec.shape[0]))
    v_length = np.linalg.norm(vec)
    counter = 0
    vec = vec/v_length
    while counter < n:
        u = np.random.uniform(-1, 1, vec.shape[0])
        u = u/np.linalg.norm(u)
        cos_theta = np.dot(u, vec) / (np.linalg.norm(u))
        if cos_theta >= math.cos(radian_e):
            e_neighbors = np.vstack((e_neighbors, u))
            counter += 1
    return e_neighbors


def get_measures_for_vectors(vectors, bw, k,  batch, epochs, repeat):
    measures = np.zeros((0, 4))

    for i, u in enumerate(vectors):
        start = time.time()
        data_p = util.project_points(data[:, :-1], u)
        data_p = np.hstack((data_p, data[:, -1].reshape((-1,1))))
        nn_acc = NN.run_model_test_train_split(data_p, test_size=0.3, repeat=repeat, batch=batch, epochs=epochs)
        nn_acc_avg = np.mean(nn_acc)
        kl_div = hd.calculate_kl_div_measure(data_p, bw=bw)
        moransi = hd.moran(data_p, k=k)
        measures = np.vstack((measures, np.array([nn_acc_avg, kl_div, moransi])))
        e = int(time.time() - start)
        print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
        print(i)

    return measures


def get_spearmanr(data1, data2):
    corr, _ = spearmanr(data1, data2)
    return corr


def get_pearsonr(data1, data2):
    corr, _ = pearsonr(data1, data2)
    return corr


def get_close_vectors_indx(vector_list, vec, epsilon):
    radian_e = math.radians(epsilon)
    vec = vec / np.linalg.norm(vec)
    indx = []
    for i, u in enumerate(vector_list):
        cos_theta = np.dot(u, vec) / (np.linalg.norm(u))
        if cos_theta >= math.cos(radian_e):
            indx.append(i)
    return indx


def get_angles(vector_list, vec):
    a = []
    vec = vec / np.linalg.norm(vec)
    for v in vector_list:
        cos = np.dot(v, vec) / (np.linalg.norm(v))
        theta = math.acos(np.abs(cos))
        a.append(theta)
    return a


if __name__=='__main__':

    C = [0.1, 10, 30, 50, 100, 200,
         # 300, 400, 500, 600, 700, 800, 900,
         1000]

    real_world = 'pointsets/real'
    heart_dir = real_world + '/' + 'heart_failure_clinical_records_dataset.csv'
    bank_dir = real_world + '/' + 'data_banknote_authentication.csv'
    adult_dir = 'pointsets/real/adult.npy'

    data_csv = pd.read_csv(heart_dir)
    labels = data_csv['DEATH_EVENT']
    data_csv = data_csv.drop(['time'], axis=1)
    data = data_csv.values
    scalar = MinMaxScaler()
    data = scalar.fit_transform(data)

    data_csv = pd.read_csv(bank_dir, header=None)
    data = data_csv.values
    scalar = MinMaxScaler()
    data = scalar.fit_transform(data)
    vec_svm = get_svm_separator(data, cv=10)[0]
    vec_svm = vec_svm/np.linalg.norm(vec_svm)

    e_neighbors = find_epsilon_close_vectors(vec_svm, 20, 1000)
    e_neighbors = np.vstack((vec_svm, e_neighbors))

    measures = np.zeros((0, 4))
    vectors = np.zeros((0, data.shape[1]-1))

    all = all[all[:, 5].argsort()]
    m_kl_div = np.zeros((0, 4))
    for i, u in enumerate(all[:10, :4]):
        start = time.time()
        data_p = util.project_points(data[:, :-1], u)
        data_p = np.hstack((data_p, data[:, -1].reshape((-1,1))))
        nn_acc = NN.run_model_test_train_split(data_p, test_size=0.3, repeat=100, batch=256, epochs=50)
        nn_acc_avg, nn_acc_std = np.mean(nn_acc), np.std(nn_acc)
        kl_div = hd.calculate_kl_div_measure(data_p, bw=0.06)
        moransi = hd.moran(data_p, k=15)
        m_kl_div = np.vstack((m_kl_div, np.array([nn_acc_avg, nn_acc_std, kl_div, moransi])))
        e = int(time.time() - start)
        print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
        print(i)


    repeat = 10
    batch = 256
    epochs = 50
    bw = 0.06
    k = 10
    n = 1980
    measures = np.zeros((0, 3))
    vectors = np.zeros((0, data.shape[1] - 1))


    for i in range(n):
        start = time.time()
        data_p, v = randomly_project_data(data)
        vectors = np.vstack((vectors, v))
        nn_acc = NN.run_model_test_train_split(data_p, test_size=0.3, repeat=repeat, batch=batch, epochs=epochs)
        nn_acc_avg = np.mean(nn_acc)
        kl_div = hd.calculate_kl_div_measure(data_p, bw=bw)
        moransi = hd.moran(data_p, k=k)
        measures = np.vstack((measures, np.array([nn_acc_avg, kl_div, moransi])))
        e = int(time.time() - start)
        print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
        print(i)

    # a2 = []
    b2 =[]
    radian_e = math.radians(20)
    for v in all2[:, :4]:
        cos = np.dot(v, vec_svm) / (np.linalg.norm(v))
        cos = np.abs(cos)
        b2.append(cos >= math.cos(radian_e))
        # theta = math.acos(cos)
        # a2.append(theta)

import seaborn as sns
import matplotlib.pyplot as plt
rand_measure = np.load('random_projection_measures.npy')
rand_vecs = np.load('random_projection_vectors_real_data.npy')
en_measure = np.load('real_measures_epsilon_neighborhood.npy')
en_vecs = np.load('vectors_epsilon_neighborhood.npy')
vec_svm = np.load('svm_vector.npy')


data_p = util.project_points(data[:, :-1], vec_svm)
data_p = np.hstack((data_p, data[:, -1].reshape((-1,1))))
kl_div = hd.calculate_kl_div_measure(data_p, bw=0.06)
moransi = hd.moran(data_p, k=15)


palette = sns.color_palette("tab10")

# m, mi, sym = 'KL Divergence', 2, 'kldiv'
# m, mi, sym = "Moran's I", 3, 'moransi'
sns.set(rc={'figure.figsize': (12, 5)})
sns.set_style(style='white')
ls, fs = 8, 13

fig, axs = plt.subplots(1, 2)
# epsilon neighborhood
# sns.regplot(x=en_measure[:, 0], y=en_measure[:, mi], line_kws={'color' : palette[1]})

axs[0].scatter(x=rand_measure[:, 0], y=rand_measure[:, 1], color=palette[0], alpha=0.5)
axs[0].scatter(x=en_measure[:, 0], y=en_measure[:, 2], color=palette[3], alpha=0.5)
axs[0].set_title('KL Divergence')
axs[0].title.set_size(fs)


axs[1].scatter(x=rand_measure[:, 0], y=rand_measure[:, 2], color=palette[0], alpha=0.5)
axs[1].scatter(x=en_measure[:, 0], y=en_measure[:, 3], color=palette[3], alpha=0.5)
axs[1].set_title("Moran's I")
axs[1].title.set_size(fs)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("NN Accuracy(%)", fontsize=fs)

plt.savefig('real_measures.pdf'.format(), dpi=200)
plt.show()
#
# fig, axs = plt.subplots(1, 2)
# # random projections
# axs[0].scatter(x=rand_measure[:, 0], y=rand_measure[:, 1], color=palette[0], alpha=0.7)
# axs[0].set_title('KL Divergence')
# axs[1].scatter(x=rand_measure[:, 0], y=rand_measure[:, 2], color=palette[0], alpha=0.7)
# axs[1].set_title("Moran's I")
#
# fig.add_subplot(111, frameon=False)
# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# plt.xlabel("NN Accuracy(%)")
# plt.savefig('real_random-nn.pdf'.format(), dpi=200)
# plt.show()



all_en = np.zeros((en_vecs.shape[0], en_vecs.shape[1] + en_measure.shape[1] + 1))
all_en[:, :-1] = np.hstack((en_vecs, en_measure))

all = np.zeros((rand_vecs.shape[0], rand_vecs.shape[1] + rand_measure.shape[1] + 1))
all[:, :-1] = np.hstack((rand_vecs, rand_measure))

for i, v in enumerate(all_en[:, :4]):
    cos = np.dot(v, vec_svm) / (np.linalg.norm(v))
    cos = np.abs(cos)
    all_en[i, -1] = math.acos(cos)

for i, v in enumerate(all[:, :4]):
    cos = np.dot(v, vec_svm) / (np.linalg.norm(v))
    cos = np.abs(cos)
    all[i, -1] = math.acos(cos)
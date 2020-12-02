import random, math, os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import util
from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss
from sklearn.model_selection import GridSearchCV


def generate_random_unit_vector():
    phi = math.radians(360 * random.uniform(0, 1))
    theta = math.radians(360 * random.uniform(0,1))
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)
    return np.array([x, y, z])


def remove_projection(v, p):
    scale = np.dot(p, v)/(np.linalg.norm(p)**2)
    return v - scale*p


def is_linearly_separable(data, labels):
    clf = svm.LinearSVC(C=2**20)
    clf.fit(data, labels)
    acc = accuracy_score(labels, clf.predict(data))
    return acc >= 0.99


def get_labels(data, feature):
    return np.array(data[feature])


def project_dataset(data, projection_vector):
    projected = np.empty_like(data)
    i = 0
    for point in data:
        projected[i] = remove_projection(point, projection_vector)
        i = i+1
    return projected


def generate_batch_vectors(n = 500):
    vectors = np.empty((0, 3))
    for i in range(n):
        vectors = np.append(vectors,[generate_random_unit_vector()], axis=0)
    return vectors


def process_batch(data, batch, sep_labels, mix_labels):
    remaining = np.empty((0,3))
    for i in range(batch.shape[0]):
        data_proj = project_dataset(data, batch[i])
        b1 = is_linearly_separable(data_proj, sep_labels)
        b2 = is_linearly_separable(data_proj, mix_labels)
        if b1 and not b2:
            remaining = np.append(remaining , [batch[i]], axis = 0)
    return remaining


def get_hinge_loss_mixedness(projected_data, labels):
    X, Y = projected_data, labels
    clf = LinearSVC(C=10)
    clf.fit(X, Y)
    return hinge_loss(Y, clf.decision_function(X))


def fast_process_batch(data, batch, sep_labels, mix_labels):
    remaining = np.empty((0,3))
    mixedness = []
    for vector in range(batch):
        data_proj = util.project_points_2D(data, vector)
        b1 = util.is_linearly_separable(data_proj, sep_labels)
        b2 = util.is_linearly_separable(data_proj, mix_labels)
        if b1 and not b2:
            remaining = np.append(remaining, vector, axis=0)
            mixedness.append(util.get_mixedness(data_proj, mix_labels))
    return remaining, mixedness


def process_batch_vectors_hinge(data, batch, sep_labels, mix_labels):
    remaining = np.empty((0,3))
    mixedness = []
    for vector in batch:
        data_proj = util.project_points(data, vector)
        b1 = util.is_linearly_separable(data_proj, sep_labels)
        b2 = util.is_linearly_separable(data_proj, mix_labels)
        if b1 and not b2:
            remaining = np.vstack((remaining, vector))
            mixedness.append(get_hinge_loss_mixedness(data_proj, mix_labels))
    return remaining, mixedness


def all_projections(epoch=200):
    for i in range(epoch):
        batch = generate_batch_vectors()


def plot_vectors(vecs, ax, c= 'b'):
    for i in range(vecs.shape[0]):
        ax.scatter(vecs[i][0], vecs[i][1], vecs[i][2], c, marker='.')


def save_output(output):
    i = 0
    while os.path.exists("output%s.npy" % i):
        i += 1
    np.save("output%s.npy" % i, output)


def stereographic_projection_x1(vecs):
    projected=np.empty((0,3))
    for i in range(vecs.shape[0]):
        t = 1 / vecs[i][0]
        p = np.array([t * vecs[i][0], t * vecs[i][1], t * vecs[i][2]])
        projected = np.append(projected, [p], axis=0)
    return projected


def plot_data(data, sep_labels, mix_labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    colors, markers = util.get_point_style(sep_labels, mix_labels)
    util.plot_points_3D(data, ax, c=colors, marker=markers)

    plt.show()


if __name__ == '__main__':
    # n_iter = input('Please enter number of samplings (will be separated in batches of 200):\n')
    # n_iter = int(n_iter)
    batch_size = 2000
    epochs = 100

    points = pd.read_csv("data.csv")
    sep_f, mix_f = 'f_1', 'f_2'
    data = np.array(points.iloc[:, :3])
    sep_labels = get_labels(points, sep_f)
    mix_labels = get_labels(points, mix_f)
    counter = 0
    read = False
    file = 'counter_example.npy'

    all = np.empty((0, 3))
    mixedness= np.empty((0, 1))
    if not read:
        for i in range(epochs):
            batch = generate_batch_vectors(batch_size)
            vects, measures = process_batch_vectors_hinge(data, batch, sep_labels, mix_labels)
            all = np.append(all, vects, axis=0)
            mixedness = np.vstack((mixedness, np.array(measures)))
            counter = counter + 1
            print('Epoch {} out of {}'.format(counter, epochs))
        np.save('counter_example.npy', all)
    else:
        all = np.load(file)
        for vec in all:
            data_proj = util.project_points(data, vec)
            mixedness= np.vstack((mixedness, util.get_mixedness(data_proj, mix_labels)))

    print("The number of eligible vectors is %d" % all.shape[0])

    print("Plotting 3D and 2D illustrations.")
    projected = util.stereographic_projection_x1(all[:, :-1])



    fig_2d = plt.figure(2)
    ax_2d = fig_2d.add_subplot(111)
    util.visualise_2D(projected[:, 1:3], ax_2d, cmap_name='gist_heat', c=mixedness)
    ax_2d.set_xlim(-0.5, 1.5)
    ax_2d.set_ylim(-0.5, 1.5)
    plt.show()

    # plot_data(data, sep_labels, mix_labels)

####################################
# dp = util.project_points_2D(data, np.array([[-0.82674968,  0.34091223, -0.44750846]]))
# g1 = dp[sep_labels == 1]
# g2 = dp[sep_labels == 0]
# plt.plot(g1[:, 0], g1[:, 1], 'o')
# plt.plot(g2[:, 0], g2[:, 1], 'ro')
#
# plt.show()
#
# g1 = dp[mix_labels == 1]
# g2 = dp[mix_labels == 0]
# plt.plot(g1[:, 0], g1[:, 1], 'o')
# plt.plot(g2[:, 0], g2[:, 1], 'ro')
# plt.show()

n = 1000
x = np.array([1]*n).reshape((-1,1))
extras_1 = np.hstack((x, np.random.uniform(0, 0.86, n).reshape((-1, 1)), np.random.uniform(0.85, 1.15, n).reshape((-1,1))))
extras_2 = np.hstack((x, np.random.uniform(0.85, 1.1, n).reshape((-1, 1)), np.random.uniform(0, 0.86, n).reshape((-1,1))))
extras_3 = np.hstack((x, np.random.uniform(0.85, 1.1, n).reshape((-1, 1)), np.random.uniform(0.85, 1.1, n).reshape((-1,1))))
extras = np.vstack((extras_1, extras_2, extras_3))
v, m = process_batch_vectors_hinge(data, extras, sep_labels, mix_labels)


n = 5000
xe = np.array([1]*n).reshape((-1,1))
e = np.hstack((xe, np.random.uniform(0.97, 1.1, n).reshape((-1, 1)), np.random.uniform(0, 1, n).reshape((-1,1))))
ve, me = process_batch_vectors_hinge(data, e, sep_labels, mix_labels)

vt = np.vstack((v, ve))
mt = m + me

plt.scatter(vt[:, 1], vt[:,2], cmap='gist_heat', c=mt, marker='.', s=20)
plt.colorbar()
plt.show()

v = vt
m = mt



extras = extras.reshape((-1,2))
extras = np.hstack((ones, extras))
for u in extras:
    data_p = util.project_points(data, u)
    e_loss.append(get_hinge_loss_mixedness(data_p, mix_labels))
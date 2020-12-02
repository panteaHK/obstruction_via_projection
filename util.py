import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import Polygon
from os import  walk


def stereographic_projection_x1(vecs):
    projected=np.empty((0,3))
    for i in range(vecs.shape[0]):
        t = 1 / vecs[i][0]
        p = np.array([t * vecs[i][0], t * vecs[i][1], t * vecs[i][2]])
        projected = np.append(projected, [p], axis=0)
    return projected


def visualise_3D (data, ax, c= 'b', marker='.'):
    for point in data:
        ax.scatter(point[0], point[1], point[2], c=c, marker=marker)


def visualise_2D (data, ax, cmap_name, c= 'b', marker='.'):
    ax.scatter(data[:, 0], data[:, 1], c = c, marker=marker, cmap="gist_heat")
    # for point in data:
    #     ax.scatter(point[0], point[1], c=c, marker=marker)


def normalize(x):
    return x / np.linalg.norm(x)


def gram_schmidt(A):
    n = A.shape[0]
    A[0, :] = normalize(A[0, :])
    for i in range(1, n):
        Ai = A[i, :]
        for j in range(0, i):
            Aj = A[j, :]
            t = Ai.dot(Aj)
            Ai = Ai - t * Aj
        A[i, :] = normalize(Ai)
    return A


def generate_base(index, dim):
    arr = np.zeros((dim,), dtype='float64')
    arr[index] = 1
    return arr


def translate_to_bases(points, A):
    new_points = []
    for point in points :
        l = []
        for basis in A:
            l.append(np.dot(point, basis))
        new_points.append(l)
    return np.array(new_points, dtype='float64')


def project_points(points, projection_vector):
    dim = points.shape[1]
    A = np.empty((0, dim))
    A = np.vstack((A, projection_vector))

    for i in range(1, dim):
        A = np.vstack((A, generate_base(i, dim)))

    gs = gram_schmidt(A)
    new_points = translate_to_bases(points, gs[1:, :])
    return new_points


def is_linearly_separable(data, labels):
    b = labels == 1
    g_1 = data[b]
    g_2 = data[~b]

    h_1 = ConvexHull(data[b])
    h_2 = ConvexHull(data[~b])

    p_1 = Polygon(g_1[h_1.vertices, :])
    p_2 = Polygon(g_2[h_2.vertices, :])

    return not p_1.intersects(p_2)


def get_point_style (sep_labels, mix_labels):
    markers = []
    colors = []

    for label in sep_labels:
        if label == 1:
            colors.append('red')
        else:
            colors.append('blue')

    for label in mix_labels:
        if label == 1:
            markers.append('x')
        else:
            markers.append('o')

    return colors, markers


def plot_points_3D (data, ax, c, marker):
    for i in range(data.shape[0]):
        ax.scatter(data[i, 0], data[i, 1], data[i, 2], c=c[i], marker=marker[i])


def get_mixedness(projected_data, mix_labels):
    b = mix_labels == 1
    g_1 = projected_data[b]
    g_2 = projected_data[~b]

    h_1 = ConvexHull(g_1)
    h_2 = ConvexHull(g_2)

    p_1 = Polygon(g_1[h_1.vertices, :])
    p_2 = Polygon(g_2[h_2.vertices, :])

    return p_1.intersection(p_2).area


def plot_points_2D(data, plt, outputfile, format='png', dpi=800):
    color = ['red' if l == 0 else 'blue' for l in data['label']]
    plt.scatter(data['x'], data['y'], color=color, marker='.')
    plt.savefig(outputfile, format=format, dpi=dpi)
    plt.show()


def plot_points_2d_np(data, plt, outputfile, dpi=200):
    color = ['blue' if l == 0 else 'red' for l in data[:, -1]]
    plt.scatter(data[:, 0], data[:, 1], color=color, marker='.')
    plt.savefig(outputfile, dpi=dpi)
    plt.show()


def get_file_names(dir):
    f = []
    for (dirpath, dirnames, filenames) in walk(dir):
        f.extend(filenames)
        break
    return f


def get_colors(data):
    color = ['blue' if l == 0 else 'red' for l in data[:, -1]]
    return color


def sort_by_column(arr, i):
    return arr[arr[:,i].argsort()]
import os
import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def accumulate_data(name):
    i = 0
    data = np.empty((0,3))
    file_name =name+str(i)+'.npy'
    while os.path.exists(file_name):
        data = np.append(data, np.load(file_name), axis=0)
        i = i + 1
        file_name = name+str(i)+'.npy'
    np.save('results/all_data.npy', data)
    return data

def visualise_3D (data, ax, c= 'b'):
    for point in data:
        ax.scatter(point[0], point[1], point[2], c=c, marker='.')


def visualise_2D (data, ax, c= 'b'):
    for point in data:
        ax.scatter(point[0], point[1], c=c, marker='.')


def stereographic_projection_x1(vecs):
    projected = np.empty((0,3))
    for i in range(vecs.shape[0]):
        t = 1 / vecs[i][0]
        p = np.array([t * vecs[i][0], t * vecs[i][1], t * vecs[i][2]])
        projected = np.append( projected, [p] , axis=0)
    return projected

if __name__ == "__main__":
    fig_3d = plt.figure(1)
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # data = accumulate_data('results\output')
    data = np.load('results/all_data.npy')

    projected = stereographic_projection_x1(data)
    visualise_3D(data, ax_3d, c='r')
    visualise_3D(projected, ax_3d, c='b')
    ax_3d.set_xlim3d(-3, 3)
    ax_3d.set_ylim3d(-3, 3)
    ax_3d.set_zlim3d(-3, 3)
    plt.show()

    fig_2d = plt.figure(2)
    ax_2d = fig_2d.add_subplot(111)
    visualise_2D(projected[:, 1:3], ax_2d)
    ax_2d.set_xlim(-5, 5)
    ax_2d.set_ylim(-5, 5)
    plt.show()
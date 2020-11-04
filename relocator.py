import numpy as np
import pandas as pd
import util
import matplotlib.pyplot as plt


def move_horizontally_towards(d1, d2, e):
    max_d1 = find_point_maximum_x(d1)
    max_d2 = find_point_maximum_x(d2)
    maximum = max_d1
    if max_d1[0] > max_d2[0]:
        right, left = d1, d2
    else:
        right, left = d2, d1
        maximum = max_d2
    minimum = find_point_minimum_x(left)
    v_right = (maximum - minimum) * 0.5
    v_left = (minimum - maximum) * 0.5
    v_right[2], v_left[2] = 0, 0

    left = translate(left, (v_right * e ))
    print
    right = translate(right, (v_left * e))

    return right, left



def find_point_maximum_x(d):
    max=d[0]
    for point in d:
        if point[0] > max[0]:
            max = point
    return max


def find_point_minimum_x(d):
    min=d[0]
    for point in d:
        if point[0] > min[0]:
            min = point
    return min


def translate(points, vec):
    l = []
    for point in points:
        l.append((point + vec))
    return np.array(l)


def maximum_outer_circle_points(r_s, r_b, n_points, color=1):
    points = np.empty((n_points, 3))
    for i in range(n_points):
        x, y = np.random.uniform(-r_b, r_b), np.random.uniform(-r_b, r_b)
        r2 = x**2+ y**2
        while r2 < r_s**2 or r2 > r_b**2:
            x, y = np.random.uniform(-r_b, r_b), np.random.uniform(-r_b, r_b)
            r2 = x ** 2 + y ** 2
        points[i, 0], points[i, 1], points[i, 2] = x, y, color
    return points







data =  pd.read_csv('pointsets/pointset8.csv')
data = np.array(data)
d1 = data[data[:, -1] == 0]
d2 = data[data[:, -1] == 1]
# fig, axs = plt.subplots(1,5)
for i in range(1,6):
    right, left = move_horizontally_towards(d1, d2, i * 0.2)
    d = np.vstack((right, left))
    color = ['red' if l == 0 else 'blue' for l in d[:, -1]]
    # axs[i-1].set_aspect(aspect='equal', adjustable='box')
    plt.scatter(d[:, 0], d[:, 1], color=color, marker='o')
    # plt.title('({}%)'.format(i*20))
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        # top=False,  # ticks along the top edge are off
        left=False,
        labelbottom=False,
        labelleft=False)  # labels along the bottom edge are off
    plt.xlim((-4, 14.2))
    plt.ylim((-6, 6))
    plt.savefig('intertwining{}.png'.format(i), dpi=800)
    plt.show()

# if __name__ == '__main__':
    #grow_inner_circle(4, 8, 10, color=1)
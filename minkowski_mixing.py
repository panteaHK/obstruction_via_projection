import sampling_points as sp
import numpy as np
import util
import matplotlib.pyplot as plt


def minkowski_mixing_inner_circles(r_inner, r_outer, n_samples, image_output=False, trial_range=(0, 1)):
    blue_step = r_inner/10
    for i in range(0,11):
        for j in range(*trial_range):
            red = sp.make_circle(n_samples=n_samples, radius=r_inner, color=1)
            blue = sp.make_2d_donut(n_samples, r_inner, r_outer, color=0)
            red_mink = sp.make_circle(n_samples= n_samples, radius=blue_step*i, color=0)
            blue_mink = sp.make_circle(n_samples=n_samples, radius=blue_step*i, color=0)
            data = np.vstack((red+red_mink, blue_mink+blue))
            np.save('pointsets/minkowski_inner_circle/inner_circle_minkowski_{}_{}.npy'.format(i*10, j), data)
        if image_output:
            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                # top=False,  # ticks along the top edge are off
                left=False,
                labelbottom=False,
                labelleft=False)
            util.plot_points_2d_np(data, plt, 'pointsets/minkowski_inner_circle/inner_circle_minkowski_{}.png'.format(i*10))


def minkowski_mixing_checkerboards(dim, n_samples, image_output=False, trial_range=(0, 1)):
    unit = dim/8
    centers = sp.get_centers(dim)
    squares = sp.get_checkerboard_square_coordinates(centers, unit)
    n = int(n_samples/64)
    for i in range(11):
        for j in range(*trial_range):
            noise = sp.make_circle(radius=unit*i/10, n_samples=n*64, color=0)
            data = sp.make_checker_squares(squares, n, dim)
            data = data + noise
            data = sp.cut_off_points(data, (0,dim), (0,dim))
            np.save('pointsets/minkowski_checkerboard/checkerboard_minkowski_{}_{}.npy'.format(i * 10, j), data)
        if image_output:
            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                left=False,
                labelbottom=False,
                labelleft=False)
            util.plot_points_2d_np(data, plt, 'pointsets/minkowski_checkerboard/checkerboard_minkowski_{}.png'.format(i*10))


def minkowski_mixing_cross(x1,x2,x3,x4,y1,y2,y3,y4, n_samples, image_output=False, trial_range=(0, 1)):
    for i in range(0,11):
        for k in range(*trial_range):
            reds = np.empty((0,3))
            blues = np.empty((0,3))
            for j in range(int(n_samples/2)):
                (ux, dx) = np.random.uniform(x2, x3 , 2)
                uy = np.random.uniform(y3, y4)
                dy = np.random.uniform(y1, y2)
                (ry, ly) = np.random.uniform(y2, y3, 2)
                lx = np.random.uniform(x1, x2)
                rx = np.random.uniform(x3, x4)
                reds = np.vstack((reds, np.array([[ux,uy,1], [dx,dy,1]])))
                blues = np.vstack((blues, np.array([[rx, ry, 0], [lx, ly, 0]])))
            noise = sp.make_circle(n_samples=int(n_samples/2)*4, radius=(x3-x2)/10*i, color=0)
            data = np.vstack((reds, blues))
            data = data + noise
            np.save('pointsets/minkowski_cross/cross_minkowski_{}_{}.npy'.format(i*10, k), data)
        if image_output:
            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                # top=False,  # ticks along the top edge are off
                left=False,
                labelbottom=False,
                labelleft=False)
            util.plot_points_2d_np(data, plt, 'pointsets/minkowski_cross/cross_minkowski_{}.png'.format(i*10))


def minkowski_mixing_S(n_samples, noise=0.05, image_output=False, trial_range=(0, 1)):
    for i in range(11):
        for j in range(*trial_range):
            s = sp.make_s_dataset(n_samples, noise)
            n = sp.make_circle(n_samples=n_samples, radius=i/15)
            data = s + n
            np.save('pointsets/minkowski_s_curve/s_curve_minkowski_{}_{}.npy'.format(i*10, j), data)
        if image_output:
            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                left=False,
                labelbottom=False,
                labelleft=False)
            util.plot_points_2d_np(data, plt, 'pointsets/minkowski_s_curve/s_curve_minkowski_{}.png'.format(i*10))


def minkowski_mixing_crossing_recs(x_start, x_end, y_start, y_end, n_samples, image_output=False, trial_range=(0, 1)):
    right_step = (x_end[1] - x_start[1])/20
    left_step = (x_start[0] - x_end[0])/20
    red_step = (right_step + left_step)/2
    up_step = (y_end[1] - y_start[1])/ 20
    down_step = (y_start[0] - y_end[0])/20
    in_blue_step = (x_start[1] - x_start[0])/40
    blue_step = (up_step+down_step+in_blue_step)/3
    w, h = x_end[1], y_end[1]
    for i in range(11):
        for j in range(*trial_range):
            reds = sp.make_rectangle_lim(limx=(x_start[0], x_start[1]), limy=(0,h), n_samples=n_samples, color=1)
            red_noise = sp.make_circle(n_samples= n_samples, radius=red_step*i)
            blues1 = sp.make_rectangle_lim(limx=(0,x_start[0]), limy=(y_start[0], y_start[1]), n_samples=n_samples, color=0)
            blues2 = sp.make_rectangle_lim(limx=(x_start[1], w), limy=(y_start[0], y_start[1]), n_samples=n_samples,color=0)
            blue_noise = sp.make_circle(n_samples=2*n_samples, radius=blue_step*i)
            blues = np.vstack((blues1, blues2))
            blues = blues + blue_noise
            reds = reds + red_noise
            data = np.vstack((blues,reds))
            np.save('pointsets/minkowski_crossing_recs/crossing_recs_minkowski_{}_{}.npy'.format(i * 10, j), data)
        if image_output:
            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                # top=False,  # ticks along the top edge are off
                left=False,
                labelbottom=False,
                labelleft=False)
            util.plot_points_2d_np(data, plt, 'pointsets/minkowski_crossing_recs/crossing_recs_minkowski_{}.png'.format(i*10))


def minkowski_mixing_moons(n_samples, noise, image_output=False, trial_range=(0, 1)):
    for i in range(0,11):
        for j in range(*trial_range):
            data = sp.make_moons(n_samples, noise)
            reds = data[data[:, -1] == 1]
            blues = data[data[:, -1] == 0]
            minr = np.min(reds[:,1])
            minb = np.min(blues[:,1])
            if minr > minb:
                for p in data:
                    p[2] = (p[2] + 1) % 2
            minkowski_noise = sp.make_circle(n_samples=n_samples, radius=0.05*i)
            data = data + minkowski_noise
            np.save('pointsets/minkowski_moons/moons_minkowski_{}_{}.npy'.format(i*10, j), data)
        if image_output:
            plt.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                left=False,
                labelbottom=False,
                labelleft=False)
            util.plot_points_2d_np(data, plt, 'pointsets/minkowski_moons/moons_minkowski_{}.png'.format(i*10))


def generate_minkowski_mixings():
    minkowski_mixing_checkerboards(1, 5000, True, (0, 11))
    minkowski_mixing_checkerboards(8, 5000, True, (0, 11))
    minkowski_mixing_cross(0, 3, 4, 7, 0, 3, 4, 7, 1400, True, (0, 11))
    minkowski_mixing_S(1500, 0.05, True, (0, 11))
    minkowski_mixing_crossing_recs((1.75, 2.25), (0, 4), (1.5, 6.5), (0, 8), 1500, True, (0, 11))
    minkowski_mixing_moons(2000, 0.05, True, (0, 11))

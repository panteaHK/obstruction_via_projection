import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import util
import relocator
seed = 100


def randomize_data(data, percentage):
    n = int(np.round(data.shape[0] * percentage/100))
    indx = np.random.randint(0, data.shape[0], n)
    for i in indx:
        val = data[i, -1]
        data[i, -1] = (val + 1) % 2
    return data


def make_square(n_samples=500, dim=8, color=0):
    data = np.zeros((n_samples, 3))
    for i in range(n_samples):
        x, y = np.random.uniform(0, dim), np.random.uniform(0, dim)
        data[i, 0], data[i, 1] , data[i, 2] = x, y, color
    return data


def make_rectangle(shape, n_samples=500, color=0):
    data = np.zeros((n_samples,3))
    for i in range(n_samples):
        x, y = np.random.uniform(0, shape[0]), np.random.uniform(0, shape[1])
        data[i, 0], data[i, 1], data[i, 2] = x, y, color
    return data


def make_rectangle_lim(limx, limy, n_samples=500, color=0):
    data = np.zeros((n_samples,3))
    for i in range(n_samples):
        (x, y) = np.random.uniform(limx[0], limx[1], 2)
        data[i, 0], data[i, 1], data[i, 2] = x, y, color
    return data


def make_circle(center=np.array([0,0,0]), n_samples=500, radius=4, color=0):
    data = np.zeros((n_samples, 3))
    for i in range(n_samples):
        x, y = np.random.uniform(-radius, radius), np.random.uniform(-radius, radius)
        while x**2 + y**2 > radius**2:
            x, y = np.random.uniform(-radius, radius), np.random.uniform(-radius, radius)
        data[i, 0], data[i, 1], data[i, 2] = x, y, color
    data = relocator.translate(data, center)
    return data


def make_2d_donut(n_samples, r_i, r_o, color=0):
    data = np.zeros((n_samples, 3))
    for i in range(n_samples):
        x, y = np.random.uniform(-r_o, r_o), np.random.uniform(-r_o, r_o)
        dist2 = x ** 2 + y ** 2
        ro2, ri2 = r_o**2, r_i**2
        while (dist2 > ro2) or (dist2 < ri2):
            x, y = np.random.uniform(-r_o, r_o), np.random.uniform(-r_o, r_o)
            dist2 = x ** 2 + y ** 2
        data[i, 0], data[i, 1], data[i, 2] = x, y, color
    return data


def make_s_curve(n_samples=350, noise=0.05):
    data = np.zeros((n_samples, 3))
    data[:, :-1] = datasets.make_s_curve(n_samples=n_samples, noise=noise)[0][:, [0,2]]
    return data


def make_s_dataset(n_samples=1000, noise=0.05):
    data = make_s_curve(n_samples, noise)
    for i in range(data.shape[0]):
        p = data[i]
        x, y = p[0], p[1]
        if np.abs(y) < 1:
            data[i, -1] = 1
            if y > 0.8 and x > 0.5:
                data[i, -1] = 0
            if y < - 0.8 and x < -0.5:
                data[i, -1] = 0
    return data


def grow_rectangle_horizontally(points, x_start, x_end, percentage):
    out = np.empty((0,3))
    s = 200
    for p in points:
        if p[0] < x_start[0]:
            s = (x_start[0]-p[0]) / (x_start[0]-x_end[0])
            s *= 100
        elif p[0] > x_start[1]:
            s = (p[0] - x_start[1]) / (x_end[1] - x_start[1])
            s *= 100
        else:
            s = 0
        if s <= percentage:
            out = np.vstack((out, p))
    return out


def get_centers(dim):
    centers = np.empty((0,2))
    for i in range(8):
        for j in range(8):
            p = np.array([0.5 + i, 0.5 +j])
            centers = np.vstack((centers, p))
    return centers * (dim/8)


def get_checkerboard_square_coordinates(centers, unit):
    coordinates = []
    l = unit/2
    for c in centers:
        coordinates.append([[c[0] - l, c[1]-l], [c[0]+l, c[1]-l], [c[0]+l, c[1]+l], [c[0]-l, c[1]+l]])
    return np.array(coordinates)


def make_checkerboard(n_samples=500, dim=8):
    data = make_square(n_samples, dim)
    factor = 8/dim
    for p in data:
        x, y = np.floor(p[0] * factor), np.floor(p[1]*factor)
        if (x+y) % 2 == 0:
            p[2] = 1
    return data


def cut_off_points(data, xlim, ylim):
    new = np.empty((0,3))
    for p in data:
        x, y = p[0], p[1]
        if x >= xlim[0] and x <= xlim[1] and y >= ylim[0] and y <= ylim[1]:
            new = np.vstack((new, p))
    return new


def make_checker_squares(squares, n_samples, dim):
    coloring = [1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1]
    data = np.empty((0,3))
    for i in range(64):
        s = squares[i]
        data = np.vstack((data, make_rectangle_lim((s[0,0], s[1,0]), (s[0,1], s[2,1]), n_samples=n_samples, color=coloring[i])))
    data = cut_off_points(data, (0, dim), (0, dim))
    return data


def make_checker_squares_density(squares, density, dim):
    coloring = [1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 0, 1, 0, 1, 0,
                0, 1, 0, 1, 0, 1, 0, 1]
    data = np.empty((0,3))
    for i in range(64):
        s = squares[i]
        n_samples = (s[0, 0] - s[1,0]) * (s[0,1] - s[2,1]) * density
        n_samples = int(np.abs(np.round(n_samples)))
        data = np.vstack((data, make_rectangle_lim((s[0,0], s[1,0]), (s[0,1], s[2,1]), n_samples=n_samples, color=coloring[i])))
    data = cut_off_points(data, (0, dim), (0, dim))
    return data


def geometric_mixing_checkerboards_fixed(dim, n_samples):
    unit = dim/8
    centers = get_centers(dim)
    square_n_samples = int(np.round(n_samples/64))
    for i in range(11):
        squares = get_checkerboard_square_coordinates(centers, unit + (unit*i/10))
        data = make_checker_squares(squares, square_n_samples, dim)
        np.save('pointsets/checkerboard_gm_fixed/checkerboard_gm_fixed_{}_0.npy'.format(i * 10), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/checkerboard_gm_fixed/checkerboard_gm_fixed_{}.png'.format(i*10))


def geometric_mixing_checkerboards(dim, density):
    unit = dim/8
    centers = get_centers(dim)
    for i in range(11):
        squares = get_checkerboard_square_coordinates(centers, unit + (unit*i/10))
        data = make_checker_squares_density(squares, density, dim)
        np.save('pointsets/checkerboard_gm/checkerboard_gm_{}_0.npy'.format(i * 10), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/checkerboard_gm/checkerboard_gm_{}.png'.format(i*10))


def make_crossing_rectangles(shape_horizontal, width_vertical, n_samples=125):
    a, b = shape_horizontal
    w = width_vertical
    h_v = (0, a + b)
    h_h = (a/2, b + a/2)
    c = a/2
    w_v = (c - w/2, c + w/2)
    area_h = w_v[0] * (h_h[1] - h_h[0])
    area_v = h_v[1] * w
    r0 = make_rectangle_lim((0, w_v[0]), h_h, int(np.round(n_samples * area_h)))
    r1 = make_rectangle_lim((w_v[1], a), h_h, int(np.round(n_samples * area_h)))
    r2 = make_rectangle_lim(w_v, h_v, int(np.round(n_samples * area_v)))
    r2[:, -1] = np.ones((r2.shape[0]))
    return np.vstack((r0,r1,r2))


def make_moons(n_samples=1000, noise=.05):
    data = np.zeros((n_samples, 3))
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=noise)
    clustering = DBSCAN(eps=0.1).fit(noisy_moons[0])
    data[:, :-1] = noisy_moons[0]
    data[:, -1] = clustering.labels_
    return data


def make_cross(lim1x, lim2x, lim1y, lim2y, n_samples=500):
    d0 = make_rectangle_lim(lim1x, (lim1y[1], lim2y[0]), n_samples, color=0)
    d1 = make_rectangle_lim((lim1x[1], lim2x[0]), lim2y,  n_samples, color=1)
    d2 = make_rectangle_lim(lim2x, (lim1y[1], lim2y[0]),  n_samples, color=0)
    d3 = make_rectangle_lim((lim1x[1], lim2x[0]), lim1y,  n_samples, color=1)
    return np.vstack((d1, d2, d3, d0))


def maximum_crossing_rectangles(x1,x2,x3,x4, y1,y2,y3,y4, density=30):
    r_1 = make_rectangle_lim((x1, x2), (y1, y4), n_samples=(x2-x1)*(y4-y1)*density, color=1)
    r_2 = make_rectangle_lim((x3, x4), (y1, y4), n_samples=(x4 - x3)*(y4 - y1)*density, color=1)

    b_1 = make_rectangle_lim((x1, x4), (y1, y2), n_samples=(x4-x1)*(y2-y1)*density, color=0)
    b_2 = make_rectangle_lim((x1, x4), (y3, y4), n_samples=(x4 - x1) * (y4 - y3)*density, color=0)
    return np.vstack((r_1, r_2, b_1, b_2))


def geometric_mixing_cross(x1, x2, x3, x4, y1, y2, y3, y4, density):
    y_m = (y3+y2)/2
    x_m = (x3+x2)/2
    for i in range(0,11):
        reds = np.empty((0,3))
        blues = np.empty((0,3))
        area = (x2 - (x2-x1)/10*i) - (x3 + (x4-x3)/10*i)
        area *= (y3 - (y3-y_m)/10*i) - y4
        n_samples = int(np.abs(area * density))
        for j in range(n_samples):
            (ux, dx) = np.random.uniform(x2 - (x2-x1)/10*i, x3 + (x4-x3)/10*i, 2)
            uy = np.random.uniform(y3 - (y3-y_m)/10*i, y4)
            dy = np.random.uniform(y1, y2 + (y_m-y2)/10*i)
            (ry, ly) = np.random.uniform(y2 - (y2-y1)/10*i, y3+ (y4-y3)/10*i, 2)
            lx = np.random.uniform(x1, x2 + (x_m-x2)/10*i)
            rx = np.random.uniform(x3 - (x3-x_m)/10*i, x4)
            reds = np.vstack((reds, np.array([[ux,uy,1], [dx,dy,1]])))
            blues = np.vstack((blues, np.array([[rx, ry, 0], [lx, ly, 0]])))
        data = np.vstack((reds, blues))
        np.save('pointsets/cross_gm/cross_gm_{}_0.npy'.format(i*10), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/cross_gm/cross_gm_{}.png'.format(i*10))


def geometric_mixing_cross_fixed(x1,x2,x3,x4,y1,y2,y3,y4, n_samples):
    y_m = (y3 +y2)/2
    x_m = (x3+x2)/2
    for i in range(0,11):
        reds = np.empty((0,3))
        blues = np.empty((0,3))
        for j in range(int(n_samples/2)):
            (ux, dx) = np.random.uniform(x2 - (x2-x1)/10*i, x3 + (x4-x3)/10*i, 2)
            uy = np.random.uniform(y3 - (y3-y_m)/10*i, y4)
            dy = np.random.uniform(y1, y2 + (y_m-y2)/10*i)
            (ry, ly) = np.random.uniform(y2 - (y2-y1)/10*i, y3+ (y4-y3)/10*i, 2)
            lx = np.random.uniform(x1, x2 + (x_m-x2)/10*i)
            rx = np.random.uniform(x3 - (x3-x_m)/10*i, x4)
            reds = np.vstack((reds, np.array([[ux,uy,1], [dx,dy,1]])))
            blues = np.vstack((blues, np.array([[rx, ry, 0], [lx, ly, 0]])))
        data = np.vstack((reds, blues))
        np.save('pointsets/cross_gm_fixed/cross_gm_fixed_{}_0.npy'.format(i*10), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/cross_gm_fixed/cross_gm_fixed_{}.png'.format(i*10))


def geometric_mixing_circles(c1, c2, r, n_samples):
    c1 = np.array([c1[0], c1[1], 0])
    c2 = np.array([c2[0], c2[1], 0])
    for i in range(11):
        reds = make_circle(c1, n_samples=n_samples, radius=r, color=1)
        blues = make_circle(c2, n_samples=n_samples, radius=r, color=0)

        reds = relocator.translate(reds, (c2 - c1)*0.05*i)
        blues = relocator.translate(blues, (c1 - c2) * 0.05 * i)
        data = np.vstack((reds, blues))
        np.save('pointsets/circles_gm/circles_gm_{}_0.npy'.format(i * 10), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        plt.ylim(c1[1]-2*r, c1[1]+2*r)
        plt.xlim(c1[0]-r*1.1,  c2[0]+r*1.1)
        util.plot_points_2d_np(data, plt,'pointsets/circles_gm/circles_gm_{}.png'.format(i * 10))


def make_n_dimensional_sphere(dimension, r = 1, density=500):
    sphere = np.empty((0,3))
    n = int(np.round((np.pi*r*r*r*4)/3)) * density
    for i in range(n):
        p = np.random.uniform(-r, r, dimension)
        while np.linalg.norm(p) > r**2:
            p = np.random.uniform(-r, r, dimension)
    sphere = np.vstack((sphere, p))


def geometric_mixing_inner_circles_fixed(r_inner, r_outer, n_samples):
    red_step = (r_outer - r_inner)/10
    blue_step = r_inner/10
    for i in range(0,11):
        red = make_circle(n_samples= n_samples, radius=r_inner+(i * red_step), color=1)
        blue = make_2d_donut(n_samples, r_inner - (i * blue_step), r_outer, color=0)
        data = np.vstack((red, blue))
        np.save('pointsets/inner_circle_gm_fixed/inner_circle_gm_fixed_{}_0.npy'.format(i*10), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/inner_circle_gm_fixed/inner_circle_gm_fixed_{}.png'.format(i*10))


def geometric_mixing_inner_circles(r_inner, r_outer, density):
    red_step = (r_outer - r_inner)/10
    blue_step = r_inner/10
    for i in range(0,11):
        rred = r_inner+(i * red_step)
        rsample = int(np.pi * (rred)**2 * density)
        rblue =  r_inner - (i * blue_step)
        bsample = int(np.pi * ((r_outer**2) - (rblue**2)) * density)
        red = make_circle(n_samples=rsample, radius=r_inner+(i * red_step), color=1)
        blue = make_2d_donut(bsample, r_inner - (i * blue_step), r_outer, color=0)
        data = np.vstack((red, blue))
        np.save('pointsets/inner_circle_gm/inner_circle_gm_{}_0.npy'.format(i*10), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/inner_circle_gm/inner_circle_gm_{}.png'.format(i*10))


def geometric_mixing_s_curve(n_samples, noise):
    for i in range(11):
        reds = make_s_dataset(n_samples, noise)
        blues = make_s_dataset(n_samples, noise)
        data = reds[reds[:, -1]==1]
        data = np.vstack((data, blues[blues[:,-1]==0]))
        red_extra = reds[reds[:, -1]==0]
        blue_extra = blues[blues[:, -1]==1]
        extra_up = red_extra[red_extra[:, 1] > 0]
        extra_down = red_extra[red_extra[:, 1] < 0]
        up_max = np.max(extra_up[:, 0])
        up_min = np.min(extra_up[:, 0])
        down_min = np.min(extra_down[:,0])
        down_max = np.max(extra_down[:,0])
        for p in extra_up:
            if (p[0] - up_min) / (up_max - up_min) < i/10:
                p[-1]=1
                data = np.vstack((data, p))
        for p in extra_down:
            if (down_max - p[0]) / (down_max - down_min) < i/10:
                p[-1]=1
                data = np.vstack((data, p))

        b_max = np.max(blue_extra[:,1])
        b_min = np.min(blue_extra[:,1])

        for p in blue_extra:
            if p[1] > 0:
                if (b_max - p[1]) / (b_max) < i/10:
                    p[-1] = 0
                    data = np.vstack((data, p))
            if p[1] < 0:
                if (p[1] - b_min) / (- b_min) < i / 10:
                    p[-1] = 0
                    data = np.vstack((data, p))
            # if np.abs(p[1]) < 0.001 and i ==10:
            #     p[-1] = 0
            #     data = np.vstack((data, p))
        np.save('pointsets/s_curve_gm/s_curve_gm_{}_0.npy'.format(i*10), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/s_curve_gm/s_curve_gm_{}.png'.format(i*10))


def geometric_mixing_crossing_recs_fixed(x_start, x_end, y_start, y_end, n_samples):
    right_step = (x_end[1] - x_start[1])/10
    left_step = (x_start[0] - x_end[0])/10
    up_step = (y_end[1] - y_start[1])/ 10
    down_step = (y_start[0] - y_end[0])/10
    in_blue_step = (x_start[1] - x_start[0])/20
    w, h = x_end[1], y_end[1]
    for i in range(11):
        reds = make_rectangle_lim(limx=(x_start[0] - i*left_step, x_start[1] + i*right_step), limy=(0,h), n_samples=n_samples, color=1)
        blues1 = make_rectangle_lim(limx=(0,x_start[0] + i*in_blue_step), limy=(y_start[0] - down_step*i, y_start[1]+up_step*i), n_samples=n_samples, color=0)
        blues2 = make_rectangle_lim(limx=(x_start[1] - i*in_blue_step, w), limy=(y_start[0] - down_step*i, y_start[1] + up_step*i), n_samples=n_samples,color=0)
        data = np.vstack((blues1, blues2,reds))
        np.save('pointsets/crossing_recs_gm_fixed/crossing_recs_gm_fixed_{}_0.npy'.format(i * 10), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/crossing_recs_gm_fixed/crossing_recs_gm_fixed_{}.png'.format(i*10))


def geometric_mixing_crossing_recs(x_start, x_end, y_start, y_end, density):
    right_step = (x_end[1] - x_start[1])/10
    left_step = (x_start[0] - x_end[0])/10
    up_step = (y_end[1] - y_start[1])/ 10
    down_step = (y_start[0] - y_end[0])/10
    in_blue_step = (x_start[1] - x_start[0])/20
    w, h = x_end[1], y_end[1]
    for i in range(11):
        red_area = (x_start[1] + i*right_step - (x_start[0] - i*left_step )) * h
        blue_area = (y_start[1]+up_step*i - (y_start[0] - down_step*i)) * (x_start[0] + i * in_blue_step)
        reds = make_rectangle_lim(limx=(x_start[0] - i*left_step, x_start[1] + i*right_step), limy=(0,h), n_samples=int(red_area * density), color=1)
        blues1 = make_rectangle_lim(limx=(0, x_start[0] + i * in_blue_step),
                                    limy=(y_start[0] - down_step * i, y_start[1] + up_step * i), n_samples=int(density*blue_area),
                                    color=0)
        blues2 = make_rectangle_lim(limx=(x_start[1] - i * in_blue_step, w),
                                    limy=(y_start[0] - down_step * i, y_start[1] + up_step * i), n_samples=int(density*blue_area),
                                    color=0)
        data = np.vstack((blues1, blues2,reds))
        np.save('pointsets/crossing_recs_gm/crossing_recs_gm_{}_0.npy'.format(i * 10), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/crossing_recs_gm/crossing_recs_gm_{}.png'.format(i*10))


def probabilistic_mixing_inner_circle(r_inner, r_outer, n_samples):
    for i in range(0, 8):
        rsample = int(np.pi * (r_inner) ** 2 * n_samples)
        bsample = int(np.pi * ((r_outer ** 2) - (r_inner ** 2)) * n_samples)
        red = make_circle(n_samples=rsample, radius=r_inner, color=1)
        blue = make_2d_donut(bsample, r_inner, r_outer, color=0)
        data = np.vstack((red, blue))
        data = randomize_data(data, i*10)
        np.save('pointsets/inner_circle_pm/inner_circle_pm_{}_0.npy'.format(i * 10), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/inner_circle_pm/inner_circle_pm_{}.png'.format(i * 10))


def probabilistic_mixing_checkerboard(n_samples, dim):
    for i in range(0, 80, 10):
        data = make_checkerboard(n_samples, dim)
        data = randomize_data(data, i)
        np.save('pointsets/checkerboard_pm/checkerboard_pm_{}_0.npy'.format(i), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/checkerboard_pm/checkerboard_pm_{}.png'.format(i))


def probabilistic_mixing_circles(c1, c2, r, n_samples):
    c1 = np.array([c1[0], c1[1], 0])
    c2 = np.array([c2[0], c2[1], 0])
    for i in range(0,80, 10):
        reds = make_circle(c1, n_samples=n_samples, radius=r, color=1)
        blues = make_circle(c2, n_samples=n_samples, radius=r, color=0)
        data = np.vstack((reds, blues))
        data = randomize_data(data, i)
        np.save('pointsets/circles_pm/circles_pm_{}_0.npy'.format(i), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        plt.ylim(c1[1]-2*r, c1[1]+2*r)
        plt.xlim(c1[0]-r*1.1,  c2[0]+r*1.1)
        util.plot_points_2d_np(data, plt, 'pointsets/circles_pm/circles_pm_{}.png'.format(i))


def probabilistic_mixing_cross(lim1x, lim2x, lim1y, lim2y, n_samples):
    for i in range(0,80,10):
        data = make_cross(lim1x, lim2x, lim1y, lim2y, n_samples)
        data = randomize_data(data, i)
        np.save('pointsets/cross_pm/cross_pm_{}_0.npy'.format(i), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/cross_pm/cross_pm_{}.png'.format(i))


def probabilistic_mixing_s_curve(n_samples, noise):
    for i in range(0, 80, 10):
        data = make_s_dataset(n_samples, noise)
        data = randomize_data(data, i)
        np.save('pointsets/s_curve_pm/s_curve_pm_{}_0.npy'.format(i), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/s_curve_pm/s_curve_pm_{}.png'.format(i))


def probabilistic_mixing_rectangles(x_lim_v, y_lim_v, x_lim_h, y_lim_h, n_samples):
    for i in range(0, 80, 10):
        reds = make_rectangle_lim(limx=(x_lim_v[0], x_lim_v[1]), limy=y_lim_v, n_samples=n_samples, color=1)
        blues1 = make_rectangle_lim(limx=(0, x_lim_v[0]), limy=(y_lim_h[0], y_lim_h[1]), n_samples=n_samples, color=0)
        blues2 = make_rectangle_lim(limx=(x_lim_v[1], x_lim_h[1]), limy=(y_lim_h[0], y_lim_h[1]), n_samples=n_samples,color=0)
        data = np.vstack((blues1, blues2, reds))
        data = randomize_data(data, i)
        np.save('pointsets/crossing_recs_pm/crossing_recs_pm_{}_0.npy'.format(i), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/crossing_recs_pm/crossing_recs_pm_{}.png'.format(i))


def probabilistic_mixing_moons(n_samples, noise):
    for i in range(0, 80, 10):
        data = make_moons(n_samples, noise)
        data = randomize_data(data, i)
        np.save('pointsets/moons_pm/moons_pm_{}_0.npy'.format(i), data)
        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            # top=False,  # ticks along the top edge are off
            left=False,
            labelbottom=False,
            labelleft=False)
        util.plot_points_2d_np(data, plt, 'pointsets/moons_pm/moons_pm_{}.png'.format(i))

for i in range(11):
    g = grow_rectangle_horizontally(reds, (1,1.25), (0,2.25), i *10)
    all = np.vstack((g, blues))
    np.save('pointsets/crossing_recs_grow/crossing_recs_grow_{}.npy'.format(i*10), all)
    util.plot_points_2d_np(all, plt, 'out.png')
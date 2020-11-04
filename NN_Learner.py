import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import relocator
import util

seed = 10
np.random.seed(seed)


def get_pointset_name(n):
    return 'pointsets/pointset{}.csv'.format(n)


def plot_points_2D(data, plt):
    color = color = ['red' if l == 0 else 'blue' for l in data['label']]
    plt.scatter(data['x'], data['y'], color=color, marker='.')
    plt.show()


def plot_points_2D(data, plt):
    color = color = ['red' if l == 0 else 'blue' for l in y_test]
    plt.scatter(x_test[:, 0], x_test[:,1], color=color, marker='.')
    plt.show()


def cross_validate_data_points():
    pointset_accuracy = []

    for j in range(7):
        data = pd.read_csv(get_pointset_name(j+1))
        display_pointset(data)
        data = np.array(data)
        X = data[:, :2]
        Y = data[:, -1:]
        cvscores = []
        for i in range(5):
        kfold = StratifiedKFold(n_splits=25, shuffle=True,
                                # random_state=seed
        )
        for train, test in kfold.split(X, Y):
          # create model
            model = keras.Sequential()
            model.add(keras.Input(shape=(2,)))
            model.add(layers.Dense(2, activation="relu"))
            model.add(layers.Dense(2, activation="relu"))
            model.add(layers.Dense(2, activation="relu"))
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(
                loss=keras.losses.BinaryCrossentropy(),
                 optimizer=keras.optimizers.Adam(),
                 metrics=['accuracy']
             )
            model.fit(X[train], Y[train], epochs=50, batch_size=32, verbose=0)
            scores = model.evaluate(X[test], Y[test], verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)

        pointset_accuracy.append(np.array(cvscores).T)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        pointset_accuracy = np.array(pointset_accuracy)
    print(pointset_accuracy.mean(axis=1))
    return pointset_accuracy

def test_train_validate_data_points():
    pointset_accuracy = []

    for j in range(7):
        data = pd.read_csv(get_pointset_name(2+ 1))
        display_pointset(data)
        data = np.array(data)
        cvscores = []
        for i in range(25):
            x_train, x_test, y_train, y_test = train_test_split(data[:, :2], data[:, -1:], test_size=0.2)
            # display_pointset(np.hstack((x_train,y_train)))
             # create model
            model = keras.Sequential()
            model.add(keras.Input(shape=(2,)))
            model.add(layers.Dense(4, activation="relu"))
            model.add(layers.Dense(4, activation="relu"))
            model.add(layers.Dense(4, activation="relu"))
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(
                loss=keras.losses.BinaryCrossentropy(),
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy']
            )
            model.fit(x_train, y_train, epochs=50, batch_size=16, verbose=0)
            scores = model.evaluate(x_test, y_test, verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            pred = [1 if x > 0.5 else 0 for x in model.predict(x_test)]
            color = ['blue' if x == 0 else 'red' for x in pred]
            plt.scatter(x_test[:, 0], x_test[:, 1], color=color, marker='.')
            plt.show()
            cvscores.append(scores[1] * 100)

        pointset_accuracy.append(np.array(cvscores).T)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    pointset_accuracy = np.array(pointset_accuracy)
    print(pointset_accuracy.mean(axis=1))



x_train, x_test, y_train, y_test = train_test_split(data[:, :2], data[:, -1:], test_size=0.1)
color = color = ['red' if l == 0 else 'blue' for l in y_test]
plt.scatter(x_test[:, 0], x_test[:,1], color=color, marker='.')
plt.show()

color = color = ['red' if l == 0 else 'blue' for l in y_train]
plt.scatter(x_train[:, 0], x_train[:,1], color=color, marker='.')
plt.show()

keras.utils.plot_model(model, 'Simple_model.png', show_shapes=True)
history = model.fit(x_train, y_train, batch_size=5, epochs=10, validation_split=0.2)
model.evaluate(x_test, y_test)



data1 = make_circle(n_sample=400)
data2 = make_circle(n_sample=400)
for i in range(data.shape[0]):
    r = np.linalg.norm(data[i, :])
    if r > 2:
        data[i, -1] = 1
display_pointset(data)


np.savetxt("pointsets/pointset8.csv", data, delimiter=",")


def disp(data, accs):
    data = np.array(data)
    color = ['blue' if l == 0 else 'red' for l in data[:, -1]]
    plt.scatter(data[:, 0], data[:, 1], color=color, marker='.')
    plt.title('Accuracy: %.2f%% (+/- %.2f%%)' % (np.mean(accs), np.std(accs)))
    plt.show()


for i in range(7):
    data = pd.read_csv(get_pointset_name(i+1))
    data = np.array(data)
    disp(data, pointset_accuracy[i, :])



accuracies = []
data = pd.read_csv(get_pointset_name(8))
data = np.array(data)
d1 = data[data[:, -1] ==0]
d2 = data[data[:, -1] ==1]
for j in range(0, 21):
    print("{}% mixed".format(0.05*j))
    d_1, d_2 = relocator.move_horizontally_towards(d1, d2, 0.05*j)
    data = np.vstack((d_1,d_2))
    display_pointset(data)
    X = data[:, :2]
    Y = data[:, -1:]

    cvscores = []
    kfold = StratifiedKFold(n_splits=100, shuffle=True,
                            # random_state=seed
    )
    for train, test in kfold.split(X, Y):
      # create model
        model = keras.Sequential()
        model.add(keras.Input(shape=(2,)))
        model.add(layers.Dense(2, activation="relu"))
        model.add(layers.Dense(2, activation="relu"))
        model.add(layers.Dense(2, activation="relu"))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            loss=keras.losses.BinaryCrossentropy(),
             optimizer=keras.optimizers.Adam(),
             metrics=['accuracy']
         )
        model.fit(X[train], Y[train], epochs=50, batch_size=32, verbose=0)
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    accuracies.append(cvscores)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# print(pointset_accuracy.mean(axis=1))

files = util.get_file_names('pointsets/')
files = [f for f in files if f[-3:] == 'csv']

for filename in files:
    dir = 'pointsets/{}'.format(filename)
    data = pd.read_csv(dir)
    util.plot_points_2D(data, plt, 'pointsets/vis/{}.png'.format(filename[:-4]))



files = ['inner_circles.csv','checkerboard.csv',  'pointset8.csv', 'cross.csv', 'S_curve.csv', 'crossing_recs.csv', 'moons.csv' , 'random.csv']
titles = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
fig, axs = plt.subplots(2,4)
# plt.figure(dpi=150)
counter = 0
for i in range(2):
    for j in range(4):
        dir = 'pointsets/{}'.format(files[counter])
        print(dir)
        data = pd.read_csv(dir)
        data = np.array(data)
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
        axs[i,j].title.set_text('({})'.format(titles[counter]))

        counter += 1


for ax in axs.flat:
    ax.label_outer()
plt.savefig('overview_2d_data.pdf', format='pdf', dpi=800)
plt.show()
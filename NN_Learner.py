import numpy as np
import util
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time



def get_model(D):
    model = keras.Sequential()
    model.add(keras.Input(shape=(D,)))
    model.add(layers.Dense(4*D, activation="relu", kernel_initializer='he_normal'))
    model.add(layers.Dense(4*D, activation="relu", kernel_initializer='he_normal'))
    model.add(layers.Dense(2*D, activation="relu", kernel_initializer='he_normal'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(beta_1=0.99, beta_2=0.99999),
        metrics=['accuracy']
    )
    return model


def real_model(D):
    model = keras.Sequential()
    model.add(keras.Input(shape=(D,)))
    model.add(layers.Dense(64, activation="relu", kernel_initializer='he_normal'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation="relu", kernel_initializer='he_normal'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation="relu", kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(beta_1=0.99, beta_2=0.99999),
        metrics=['accuracy']
    )
    return model


def run_real_model_test_train_split(data, test_size, repeat=10, epochs=30):
    X , Y = data[:, :-1], data[:, -1]
    D = X.shape[1]
    accs = []
    for i in range(repeat):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=100)
        model = real_model(D)
        model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=0)
        scores = model.evaluate(x_test, y_test, verbose=0)
        accs.append(scores[1]*100)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    return accs


def run_model_test_train_split(data, test_size, repeat=10, batch=128, epochs=30):
    start = time.time()
    X , Y = data[:, :-1], data[:, -1]
    D = X.shape[1]
    accs = []
    for i in range(repeat):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=100)
        model = get_model(D)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch, verbose=0)
        scores = model.evaluate(x_test, y_test, verbose=0)
        accs.append(scores[1]*100)
        # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        e = int(time.time() - start)
    print('Total time {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    return accs


def train_test_all_files(dir, test_size, repeat=5, n_trials=10, epochs=30):
    start = time.time()
    files = util.get_file_names(dir)
    counter=0
    n = len(files)
    acc_measures= np.zeros((repeat, n_trials, int(n/n_trials)))
    p = 20
    if dir[-2:] == 'pm':
        p = 10
    for f in files:
        s = time.time()
        print(f)
        data = np.load(dir+'/'+f)
        accs = run_model_test_train_split(data, test_size=test_size, repeat=repeat, epochs=epochs)
        step = int(f[:-4].split('_')[-2])
        trial = int(f[:-4].split('_')[-1])
        acc_measures[:, trial, int(step/p)] = accs
        counter += 1
        print("progress: {}/{}".format(counter, n))
        e = int(time.time() - start)
        ep = int(time.time() - s)
        print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (ep % 3600 // 60), e % 60))
        print('So far {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    name = dir.split('/')[-1] + '_' + dir.split('/')[-2]
    np.save('measures/high_dimensional/NN_accs/{}.npy'.format(name), acc_measures)
    e = int(time.time()-start)
    print('Total time {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    return acc_measures


if __name__ == '__main__':

    dataset_name = [
        'hyper_inner_balls_gm',
        'hyper_inner_balls_pm',
        'hyper_spheres_gm',
        'hyper_spheres_pm'
    ]

    Ds = ['3D', '4D', '5D', '6D', '7D', '8D', '9D', '10D']

    import os.path

    for d in Ds:
        for n in dataset_name:
            path = 'pointsets/high_dimensional/{}/{}'.format(d,n)
            measure_file = 'measures/high_dimensional/NN_accs/{}_{}.npy'.format(n,d)
            if not os.path.exists(measure_file):
                print('{}'.format(path))
                train_test_all_files(path, 0.3, epochs=20)

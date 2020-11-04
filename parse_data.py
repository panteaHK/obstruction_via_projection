import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

name = 'pointset7'

filename = 'pointsets/'+ name +'.txt'
new_name = 'pointsets/' + name + '.csv'

def convert_tikz_csv(filename, new_name):
    with open(filename, 'r') as file:
        content = file.read()
        label_1 = re.findall('qqqqff\] [(]([0-9.]*,[0-9.]*)[)] circle [(]2.5pt[)];', content)
        label_2 =  re.findall('ccqqqq\] [(]([0-9.]*,[0-9.]*)[)] circle [(]2.5pt[)];', content)
        with open(new_name, 'w') as output:
            output.write('x,y,label\n')
            for point in label_1:
                output.write(point + ',0\n')
            for point in label_2:
                output.write(point + ',1\n')


def display_pointset_file(filename):
    data = pd.read_csv(filename)
    display_pointset(data)


def display_pointset(data):
    data = np.array(data)
    color = ['blue' if l == 0 else 'red' for l in data[:, -1]]
    plt.scatter(data[:, 0], data[:, 1], color=color, marker='.')
    # plt.ylim((0,15))
    # plt.xlim((0,15))
    plt.show()


convert_tikz_csv(filename, new_name)
display_pointset_file(new_name)
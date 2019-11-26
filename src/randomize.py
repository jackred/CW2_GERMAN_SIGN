# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# author: JackRed <jackred@tuta.io>


#  import csv
from random import randint


def randomize_file(name, array):
    imgs = []
    with open(path+name, 'r') as f:
        h = f.readline()
        for i in f:
            imgs.append(i)
    for i in range(len(array)):
        invert(imgs, i, array[i])
    with open(path+'r_'+name, 'w') as f:
        f.write(h)
        for i in imgs:
            f.write(i)


def invert(a, i, j):
    tmp = a[i]
    a[i] = a[j]
    a[j] = tmp


def fisher_yates(a, s):
    for i in range(s):
        j = randint(0, i)
        invert(a, i, j)


path = '../data/'
prefix = 'y_train_smpl_'
prefix2 = 'y_test_smpl_'

extension = '.csv'
main_file = 'x_train_gr_smpl.csv'
main_file2 = 'x_test_gr_smpl.csv'

files = [main_file, prefix[0:-1]+extension]
files2 = [main_file2, prefix2[0:-1]+extension]

for i in range(10):
    files.append(prefix+str(i)+extension)
    files2.append(prefix2+str(i)+extension)

SIZE = 12660
SIZE2 = 4170
array = [i for i in range(SIZE)]
array2 = [i for i in range(SIZE2)]

fisher_yates(array, SIZE)
fisher_yates(array2, SIZE2)

for name in files:
    randomize_file(name, array)

for name in files2:
    randomize_file(name, array2)

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


def fisher_yates(a):
    for i in range(SIZE):
        j = randint(0, i)
        invert(a, i, j)


path = '../data/'
prefix = 'y_train_smpl_'
extension = '.csv'
main_file = 'x_train_gr_smpl.csv'
files = [main_file, prefix[0:-1]+extension]
for i in range(10):
    files.append(prefix+str(i)+extension)

SIZE = 12660

array = [i for i in range(SIZE)]
fisher_yates(array)
for name in files:
    randomize_file(name, array)

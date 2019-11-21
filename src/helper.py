# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import numpy as np
import preprocess
from sklearn.metrics import confusion_matrix


DELI = ','
FOLDER = '../data/random/'
DATA_FILE = 'r_x_train_gr_smpl.csv'
LABEL_FILE = 'r_y_train_smpl'
IMG_FOLDER = '../data/img/'
SEP = '_'
TEST = 'test' + SEP
LEARN = 'learn' + SEP
EXT = '.csv'
L = 8


####
# WRITE DATA
####
def write_data_to_file(name, data, fmt='%.3f', h='', cmt=''):
    # print(name)
    # print(data.shape)
    np.savetxt(name, data, delimiter=',', fmt=fmt, header=h, comments=cmt)


####
# READ DATA
####
def get_data_from_file(name, deli=DELI, dtype=float, col=None):
    if col is None:
        return np.loadtxt(name, delimiter=deli, skiprows=1, dtype=dtype)
    else:
        return np.loadtxt(name, delimiter=deli, skiprows=1, dtype=dtype,
                          usecols=col)


# give label as numpy array of integer
def get_label(sep='', i='', folder=FOLDER, label_file=LABEL_FILE,
              ext=EXT):
    folder = folder or FOLDER
    label_file = label_file or LABEL_FILE
    label = get_data_from_file((folder+label_file)+sep+str(i)+ext, dtype=int)
    return label


# give data as numpy array of integer
def get_data(folder=FOLDER, data_file=DATA_FILE, deli=DELI, col=None):
    folder = folder or FOLDER
    data_file = data_file or DATA_FILE
    return get_data_from_file(folder+data_file, deli, col=col)


# give data as numpy of string (one cell = one row)
def get_data_raw(folder=FOLDER, data_file=DATA_FILE):
        folder = folder or FOLDER
        data_file = data_file or DATA_FILE
        return np.genfromtxt(folder+data_file, dtype='str', skip_header=1)


####
# CREATE IMAGE
####
def create_image(name, d, w, h):
    write_data_to_file(name, d, fmt='%d', h='P2\n%d %d 255\n' % (w, h))


# convert an array of integer to a ppm stirng
# ex: [1.0, 2.0, 9.4] => '1\n2\n9\n'
def convert_ppm_raw(row):
    return '\n'.join([str(int(round(i))) for i in row])+'\n'


# convert csv data to ppm string
# ex: '1.0,5.0,255.0' => '1\n5\n255'
def from_csv_to_ppm_raw(row):
    return row.replace(DELI, '\n').replace('.0', '')+'\n'


def from_row_to_csv(row):
    return ', '.join(str(i) for i in row)+'\n'


def from_rows_to_csv(rows):
    return [from_row_to_csv(i) for i in rows]


def create_images_from_rows(name, rows):
    for i in range(len(rows)):
        wh = len(rows[i]) ** (1/2)
        create_image(IMG_FOLDER+'%s%i.ppm' % (name, i), rows[i], wh, wh)


####
# UTILITY
####
def print_dry(m, dry):
    if not dry:
        print(m)


####
# PREPROCESSING
####
def pre_processed_file(file_value, option, rand=0):
    if option.randomize:
        file_value = preprocess.randomize(file_value, rand)
    if option.split is not None:
        file_value, file_value_test = preprocess.split_data(file_value,
                                                            option.split)
    else:
        file_value_test = file_value
    return file_value, file_value_test


def pre_processed_data(option, rand, dry=True):
    data = get_data(folder=option.folder, data_file=option.data)
    print_dry('data loaded', dry)
    if option.contrast:
        data = preprocess.contrast_images(data, option.contrast)
        print_dry('data contrasted', dry)
    if option.equalize:
        data = preprocess.equalize_histograms(data)
        print_dry('histogram equalized', dry)
    if option.histogram:
        data = preprocess.adjust_histograms(data)
        print_dry('histogram matched', dry)
    if option.size is not None:
        data = preprocess.resize_batch(data, option.size)
        print_dry('data resized', dry)
    if option.pooling is not None:
        data = preprocess.pooling_images(data, option.pooling)
        print_dry('data resized', dry)
    if option.segment is not None:
        data = preprocess.segment_images(data, option.segment)
        print_dry('data segmented', dry)
    # if option.kmeans is not None:
    #     data = preprocess.old_segment_images(data, option.kmeans)
    #     print_dry('data kmeans-ed', dry)
    if option.filters is not None:
        data = preprocess.filter_images(data, option.filters)
        print_dry('image_filterized', dry)
    if option.binarise:
        data = preprocess.binarise_images(data)
        print_dry('data binarised', dry)
    if option.extract is not None:
        data = preprocess.extract_col(data, option.extract)
    return pre_processed_file(data, option, rand)


def pre_processed_label(option, rand, sep='', i='', dry=True):
    label = get_label(sep=sep, i=i, folder=option.folder,
                      label_file=option.label)
    print_dry('label loaded', dry)
    return pre_processed_file(label, option, rand)


####
# MATRIX PRINTING
####
def print_line_matrix(lng):
    print('-' * ((L+1) * (lng+2) + 1))


def format_string(a):
    return str(a)[:L].center(L)


def format_row(l):
    return '|'.join([format_string(i) for i in l])


def print_matrix(m, lb):
    print_line_matrix(len(lb))
    print('|' + format_string('lb\pr') + '|' + format_row(lb) + '|'
          + format_string('total') + '|')
    print_line_matrix(len(lb))
    for i in range(len(m)):
        print('|' + format_string(lb[i]) + '|' + format_row(m[i]) + '|'
              + format_string(sum(m[i])) + '|')
        print_line_matrix(len(lb))
    print('|' + format_string('total') + '|'
          + format_row(sum(m)) + '|'
          + format_string(m.sum()) + '|')
    print_line_matrix(len(lb))


# create and print confusion_matrix
def matrix_confusion(label, predicted, lb):
    matrix = confusion_matrix(label, predicted)
    #  print(100 * sum(max(matrix[:,i]) for i in range(len(matrix))) / len(label))
    #  print(list(max(matrix[:,i]) for i in range(len(matrix))))
    print_matrix(matrix, lb)

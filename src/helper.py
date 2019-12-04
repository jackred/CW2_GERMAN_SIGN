# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import numpy as np
import preprocess
from sklearn.metrics import confusion_matrix, precision_score, \
    recall_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import pydotplus
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score, classification_report, confusion_matrix
from sklearn import tree


DELI = ','
FOLDER = '../data/random/'
DATA_FILE = 'r_x_train_gr_smpl'
DATA_FILE_TEST = 'r_x_test_gr_smpl'
LABEL_FILE = 'r_y_train_smpl'
LABEL_FILE_TEST = 'r_y_test_smpl'
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
    label = get_data_from_file(folder+label_file+sep+str(i)+ext, dtype=int)
    return label


# give data as numpy array of integer
def get_data(folder=FOLDER, data_file=DATA_FILE, deli=DELI, ext=EXT, col=None):
    folder = folder or FOLDER
    data_file = data_file or DATA_FILE
    return get_data_from_file(folder+data_file+ext, deli, col=col)


# give data as numpy of string (one cell = one row)
def get_data_raw(folder=FOLDER, data_file=DATA_FILE, ext=EXT):
    folder = folder or FOLDER
    data_file = data_file or DATA_FILE
    return np.genfromtxt(folder+data_file+ext, dtype='str', skip_header=1)


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
def randomize_file(file_value, option,  rand=0):
    if option.randomize:
        file_value = preprocess.randomize(file_value, rand)
    return file_value


def split_file(file_value, option):
    if option.split is not None:
        file_value, file_value_test = preprocess.split_data(file_value,
                                                            option.split)
    else:
        file_value_test = None
    return file_value, file_value_test


def pre_processed_file(file_value, option, rand=0):
    file_value = randomize_file(file_value, option, rand)
    if option.cross_validate is not None:
        file_value_test = file_value
    else:
        file_value, file_value_test = split_file(file_value, option)
    return file_value, file_value_test


def pre_processed_data_arg(data, option, rand, dry=True):
    data = randomize_file(data, option, rand)
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
    return data


def pre_processed_label(option, rand, sep='', i='', dry=True):
    label = get_label(sep=sep, i=i, folder=option.folder,
                      label_file=option.label)
    print_dry('label  loaded', dry)
    return pre_processed_file(label, option, rand)


def pre_processed_label_test(option, rand, sep='', i='', dry=True):
    label = get_label(sep=sep, i=i, folder=option.folder,
                      label_file=option.label_test or LABEL_FILE_TEST)
    print_dry('label test loaded', dry)
    return randomize_file(label, option, rand)


def pre_processed_data(option, rand, dry=True):
    data = get_data(folder=option.folder, data_file=option.data)
    print_dry('data loaded', dry)
    return pre_processed_file(data, option, rand)


def pre_processed_data_test(option, rand, dry=True):
    data = get_data(folder=option.folder,
                    data_file=option.data_test or DATA_FILE_TEST)
    print_dry('data test loaded', dry)
    return randomize_file(data, option, rand)


def pre_processed_data_all(option, rand, dry=True):
    data_train, data_train_test = pre_processed_data(option, rand, dry)
    data_train = pre_processed_data_arg(data_train, option, rand, dry)
    if option.cross_validate is not None:
        return data_train, data_train
    if option.test:
        data_test = pre_processed_data_test(option, rand, dry)
        if data_train_test is not None:
            data_test = np.concatenate((data_test, data_train_test))
        data_test = pre_processed_data_arg(data_test, option, rand, dry)
    else:
        if data_train_test is not None:
            data_test = pre_processed_data_arg(data_train_test, option, rand,
                                               dry)
        else:
            data_test = data_train
    print(data_train.shape, data_test.shape)
    return data_train, data_test


def pre_processed_label_all(option, rand, sep='', i='', dry=True):
    label_train, label_train_test = pre_processed_label(option, rand, sep, i,
                                                        dry)
    if option.cross_validate is not None:
        return label_train, label_train
    if option.test:
        label_test = pre_processed_label_test(option, rand, sep, i, dry)
        if label_train_test is not None:
            label_test = np.concatenate((label_test, label_train_test))
    else:
        label_test = label_train \
            if label_train_test is None \
            else label_train_test
    print(label_train.shape, label_test.shape)
    return label_train, label_test


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


####
# Measure and Precision
####
def matrix_confusion(label, predicted, lb):
    matrix = confusion_matrix(label, predicted)
    #  print(100 * sum(max(matrix[:,i]) for i in range(len(matrix))) / len(label))
    #  print(list(max(matrix[:,i]) for i in range(len(matrix))))
    print_matrix(matrix, lb)


def compare_class(predicted, label):
    unique_p, counts_p = np.unique(predicted, return_counts=True)
    found = dict(zip(unique_p, counts_p))
    unique_l, counts_l = np.unique(label, return_counts=True)
    label_nb = dict(zip(unique_l, counts_l))
    print('found: ', found)
    print('label: ', label_nb)
    matrix_confusion(label, predicted, unique_l)


def compare_class_true_positive(predicted, label, specific=[]):
    u_matrix = _get_true_matrix(predicted, label)
    precision = precision_score(label, predicted, average=None)
    recall = recall_score(label, predicted, average=None)
    f1 = f1_score(label, predicted, average=None)
    for elem, p, r, f in zip(u_matrix, precision, recall, f1):
        if len(specific) == 0 or elem in specific:
            print('matrix true', elem)
            print_matrix(np.array(u_matrix[elem]), np.array([0, 1]))
            print('precision score', p)
            print('recall score', r)
            print('F measure', f)
            print()


def score_to_class(score):
    return np.array([np.argmax(i) for i in score])


def print_detail_measure(name, arr, detail=False):
    print(name, ':', sum(arr) / len(arr))
    if detail:
        for i in range(len(arr)):
            print('\t', i, ':', arr[i])


def measure(predicted, label, confidence, detail=False):
    print_detail_measure('precision score',
                         precision_score(label, predicted, average=None),
                         detail)
    print_detail_measure('recall score',
                         recall_score(label, predicted, average=None),
                         detail)
    print_detail_measure('F measure',
                         f1_score(label, predicted, average=None),
                         detail)
    scores = np.array([])
    for elem in confidence:
        scores = np.append(scores, np.amax(elem))
    true = np.array([], dtype=int)
    for exp, got in zip(label, predicted):
        val = 0
        if exp == got:
            val = 1
        true = np.append(true, int(val))
    print('ROC Area score', roc_auc_score(true, scores))


def precision(predicted, label, detail=False):
    if not detail:
        return sum(predicted == label) / len(label)
    else:
        unique = np.unique(label)
        res = []
        for i in unique:
            res.append(np.logical_and(predicted == i, label == i))
        return res


def f_measure(predicted, label, detail=False):
    score = f1_score(label, predicted, average=None)
    if not detail:
        return sum(score) / len(np.union1d(predicted, label))
    return score


def _get_true_matrix(predicted, label):
    unique_p, counts_p = np.unique(predicted, return_counts=True)
    unique_l, counts_l = np.unique(label, return_counts=True)
    matrix = confusion_matrix(label, predicted, unique_l)
    u_matrix = {}
    for elem in unique_l:
        u_matrix[elem] = [[0, 0], [0, 0]]
    for elem in u_matrix:
        for i in range(len(unique_l)):
            for j in range(len(unique_l)):
                if i == j and i == elem:
                    u_matrix[elem][0][0] = matrix[i][j]
                elif i == elem:
                    u_matrix[elem][0][1] += matrix[i][j]
                elif j == elem:
                    u_matrix[elem][1][0] += matrix[i][j]
                else:
                    u_matrix[elem][1][1] += matrix[i][j]
    return u_matrix


def true_and_false_positive(predicted, label, detail=False):
    score = recall_score(label, predicted, average=None)
    print('score', score)
    if not detail:
        res = sum(score) / len(np.union1d(predicted, label))
        return res, 1 - res
    return score, [1 - i for i in score]


def roc_score(predicted, label, confidence):
    scores = np.array([])
    for elem in confidence:
        scores = np.append(scores, np.amax(elem))
    true = np.array([], dtype=int)
    for exp, got in zip(label, predicted):
        val = 0
        if exp == got:
            val = 1
        true = np.append(true, int(val))
    return roc_auc_score(true, scores)


def all_measure(predicted, label_test):
    accs = precision(predicted, label_test)
    f_measures = f_measure(predicted, label_test)
    true_positives, false_positives = true_and_false_positive(predicted,
                                                              label_test)
    return {'accs': accs,
            'f_measures': f_measures,
            'true_positives': true_positives,
            'false_positives': false_positives}


def extract_measure(measures, key):
    return [measure[key] for measure in measures]


def extract_measures(measures):
    return {key: extract_measure(measures, key) for key in measures[0]}


def print_measures(measures):
    for k in measures:
        print(k, measures[k])


def mean_measures(measures):
    return {k[:-1]: np.mean(measures[k]) for k in measures}


####
# Cross Validation
####
def cross_validate(fn, data, label, k=10, dry=False,
                   **kwargs):
    measures = []
    datas = np.array_split(data, k)
    print(len(datas), [i.shape for i in datas])
    labels = np.array_split(label, k)
    for i in range(k):
        print('fold %d' % i)
        data_train = np.concatenate(np.concatenate((datas[:k], datas[k+1:])))
        label_train = np.concatenate(np.concatenate((labels[:k],
                                                     labels[k+1:])))
        data_test = datas[i]
        label_test = labels[i]
        predicted = fn(data_train, label_train, data_test, **kwargs)
        measures.append(all_measure(predicted, label_test))
    measures = extract_measures(measures)
    print_measures(measures)
    return mean_measures(measures)


def run_function(fn, cross, data_train, label_train, data_test, label_test,
                 **kwargs):
    if cross is not None:
        print('cross_validating')
        return cross_validate(fn, data_train, label_train, cross, **kwargs)
    else:
        print('training then testing')
        predicted = fn(data_train, label_train, data_test, **kwargs)
        measures = [all_measure(predicted, label_test)]
        measures = extract_measures(measures)
        compare_class(predicted, label_test)
        return mean_measures(measures)


COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def plot_experiment(title, x_label, to_plot):
    i = 0
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('%')
    for k in to_plot:
        plt.plot(to_plot[k], color=COLORS[i], label=k)
        i += 1
    plt.legend()
    plt.show()


def plot_experiment_server(title, x_label, to_plot):
    i = 0
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('%')
    for k in to_plot:
        plt.plot(to_plot[k], color=COLORS[i], label=k)
        i += 1
    plt.legend()
    plt.savefig(title + ".png")


def print_result(label, predicted):
    print(classification_report(label, predicted))
    print(confusion_matrix(label, predicted))


def tree_to_png(model):
    className = ["speed limit 60", "speed limit 80", "speed limit 80 lifted", "right of way at crossing",
                 "right of way in general", "give way", "stop", "no speed limit general", "turn right down",
                 "turn left down"]
    featureName = []
    for i in range(0, 2304):
        featureName.append(i)
    dot_data = tree.export_graphviz(
                model,
                out_file=None,
                feature_names=featureName,
                class_names=className
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("tree.png")

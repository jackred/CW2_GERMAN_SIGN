# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>

import helper
import arg
import numpy as np


def create_arf_file(n_attr, data, args, sep='', i=''):
    name = '%s%s%s' % ((args.name or 'arf_y_train'), sep, i)
    print('writting in %s' % name)
    header = 'relation %s\n' % name
    for i in range(n_attr):
        header += 'attribute %d numeric\n' % i
    header += ('attribute label {%s}\n' %
               ','.join(str(int(i)) for i in np.sort(np.unique(data[:, -1]))))
    header += 'data'
    helper.write_data_to_file((args.folder or helper.FOLDER)+name+'.arff',
                              data, fmt='%d', h=header, cmt='@')


def append(data, args, i='', sep=''):
    label = helper.get_label(i=i, sep=sep, folder=args.folder,
                             label_file=args.label)
    label = label.reshape(-1, 1)  # reshape for append
    data_mix = np.append(data, label, 1)
    create_arf_file(len(data[0]), data_mix, args, sep, i)
    print('label%s done' % (' ' + i))


def main():
    args = arg.preprocess_args()
    data = helper.get_data(folder=args.folder, data_file=args.data)
    print('data_loaded')
    append(data, args)
    for i in range(10):
        append(data, args, str(i), '_')


if __name__ == '__main__':
    main()

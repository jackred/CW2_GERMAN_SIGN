# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>


import arg
import helper
import random
import preprocess


def main():
    args = arg.preprocess_args()
    rand = 0
    data, _ = helper.pre_processed_data(args, rand, dry=False)
    if args.mean:
        rand = random.randint(0, 9999999)
        label, _ = helper.pre_processed_label(args, rand, dry=False)
        data = preprocess.mean_image(label, data)
    helper.create_images_from_rows((args.name or 'img'),
                                   data)


if __name__ == '__main__':
    main()

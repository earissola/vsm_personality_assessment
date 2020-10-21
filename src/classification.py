#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Implementation of 'A Vectorial Semantics Approach to Personality Assesment'
By Neuman et al. 2014. Avaialble at: https://www.nature.com/articles/srep04761
** Classification Step **
'''

from __future__ import print_function
from builtins import range
import argparse
import sys
from csv import reader, register_dialect, writer
from operator import itemgetter
from os import getcwd, listdir, makedirs
from os.path import basename, exists, isdir, isfile, join, splitext

import numpy as np
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier

__author__ = "Esteban Rissola"
__credits__ = ["Esteban Rissola"]
__version__ = "1.0.1"
__maintainer__ = "Esteban Rissola"
__email__ = "esteban.andres.rissola@usi.ch"

SEED = 10710

def classification_binary(X_tr, y_tr, X_tt, y_tt):
    # No information given in the paper about parameters - Using default ones #
    model = DecisionTreeClassifier(random_state=SEED)
    model = model.fit(X_tr, y_tr)
    y_hat = model.predict(X_tt)
    # print(classification_report(y_tt, y_hat, digits=4))
    print('Precision: %.4f' % precision_score(y_tt, y_hat, 
            average='binary'))
    print('Recall: %.4f' % recall_score(y_tt, y_hat, 
            average='binary'))
    print('F1-Score: %.4f' % f1_score(y_tt, y_hat, 
            average='binary'))

def main():
    parser = argparse.ArgumentParser(
        description = 'Neuman et. 2014 - Baseline (Classification)')

    help_msgs = []
    # Input arguments #
    help_msgs.append('training/test features path (npy)')

    # Input arguments #
    parser.add_argument('features_path', help=help_msgs[0])

    # Arguments parsing #
    args = parser.parse_args()

    # Check if input exists and is a directory. Otherwise, exit
    # No extra indentation.
    if not isdir(args.features_path):
        sys.exit('The input path does not point to a valid directory')

    # Read train/test Features #
    ft_train = np.load(join(args.features_path, 'ft_train.npy'))
    ft_test = np.load(join(args.features_path, 'ft_test.npy'))

    X_tr = ft_train[:, :-5]
    y_tr_all = ft_train[:, -5:]

    X_tt = ft_test[:, :-5]
    y_tt_all = ft_test[:, -5:]

    # Classification has to be performed for each Personality Factor. This
    # involves the execution of 5 (individual) binary classification
    # experiments #

    # * cEXT, cNEU, cAGR, cCON, cOPN

    idx2Factor = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

    for idx in range(5):
        print(' ** Personality Dimensions: %s **' % idx2Factor[idx])
        classification_binary(X_tr, y_tr_all[:, idx], X_tt, y_tt_all[:, idx])

    return 0


if __name__ == '__main__':
    main()

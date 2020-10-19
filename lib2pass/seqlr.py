from collections import defaultdict
import random
import logging
import itertools as it

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from joblib import Parallel, delayed

from .fastaparse import get_junction_seqs


log = logging.getLogger('2passtools')


SEQ_OHE = {'A': [1, 0, 0],
           'C': [0, 1, 0],
           'G': [0, 0, 1],
           'T': [0, 0, 0]}


def one_hot_sequence(seq):
    ohe = []
    for base in seq:
        try:
            ohe.append(SEQ_OHE[base])
        except KeyError:
            ohe.append(SEQ_OHE[random.choice('ACGT')])
    return np.array(ohe)


def train_and_predict(X_train, y_train, X_test):
    lr = LogisticRegression(
        solver='lbfgs', penalty='l2',
        max_iter=500, n_jobs=1)
    lr.fit(X_train, y_train)
    return lr.predict_proba(X_test)[:, 1]


def kfold_oob_prediction(X_data, y_data, n_splits, processes=1):
    idx = []
    preds = []
    kf = KFold(n_splits=n_splits, shuffle=True)
    kf_idx = list(kf.split(X_data))
    with Parallel(n_jobs=min(n_splits, processes)) as pool:
        preds = pool(
            delayed(train_and_predict)(X_data[train_idx],
                                       y_data[train_idx],
                                       X_data[test_idx])
            for train_idx, test_idx in kf_idx
        )
    test_idx = [tst for trn, tst in kf_idx]
    test_idx = np.concatenate(test_idx)
    preds = np.concatenate(preds)
    return preds[np.argsort(test_idx)]


def predict_splice_junctions_from_seq(introns, labels, fasta_fn, window_size,
                                      n_splits, processes):
    log.info(f'Fetching junction sequences from {fasta_fn}')
    (donors, donor_seqs, donor_labels,
     acceptors, acceptor_seqs, acceptor_labels) = get_junction_seqs(
        introns, labels, fasta_fn, window_size, processes
    )
    log.info(f'Identified {len(donors):d} unique donors and {len(acceptors):d} unique acceptors')
    donor_seq_ohe = np.array([one_hot_sequence(seq).ravel() for seq in donor_seqs])
    donor_labels = np.array(donor_labels)
    log.info(f'Scoring donor sequences with LR...')
    donor_preds = kfold_oob_prediction(
        donor_seq_ohe, donor_labels, n_splits, processes
    )
    donor_preds = {k: v for k, v in zip(donors, donor_preds)}
    acceptor_seq_ohe = np.array([one_hot_sequence(seq).ravel() for seq in acceptor_seqs])
    acceptor_labels = np.array(acceptor_labels)
    log.info(f'Scoring acceptor sequences with LR...')
    acceptor_preds = kfold_oob_prediction(
        acceptor_seq_ohe, acceptor_labels, n_splits, processes
    )
    acceptor_preds = {k: v for k, v in zip(acceptors, acceptor_preds)}
    donor_preds, acceptor_preds = get_donor_acceptor_preds_for_introns(
        introns, donor_preds, acceptor_preds
    )
    return donor_preds, acceptor_preds


def get_donor_acceptor_preds_for_introns(introns, donor_preds, acceptor_preds):
    intron_donor_preds = []
    intron_acceptor_preds = []
    for chrom, start, end, strand in introns:
        if strand == '+':
            donor_pos = start
            acceptor_pos = end
        else:
            donor_pos = end
            acceptor_pos = start
        intron_donor_preds.append(donor_preds[(chrom, donor_pos, strand)])
        intron_acceptor_preds.append(acceptor_preds[(chrom, acceptor_pos, strand)])
    return intron_donor_preds, intron_acceptor_preds
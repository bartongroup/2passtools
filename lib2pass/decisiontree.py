import logging
from operator import itemgetter
import re
import numpy as np

from sklearn.preprocessing import quantile_transform
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import ExtraTreesClassifier


log = logging.getLogger('2passtools')


DT1_DENOVO_FEATURES = [
    'is_canonical_motif', 'jad',
    'is_primary_donor', 'is_primary_acceptor',
    'intron_length_quantile',
]

DT2_DENOVO_FEATURES = [
    'jad', 'is_primary_donor', 'is_primary_acceptor',
    'intron_length_quantile',
    'donor_lr_score', 'acceptor_lr_score',
]


def format_feature_importances(feature_names, feature_importances, width=10):
    max_size = max(feature_importances)
    point_size = max_size / width
    pad_to = max([len(x) for x in feature_names])
    feature_importances = {fn: fi for fn, fi in zip(feature_names, feature_importances)}
    feature_importances = sorted(feature_importances.items(), key=itemgetter(1), reverse=True)
    fmt = ''
    for fn, fi in feature_importances:
        rpad = ' ' * (pad_to - len(fn))
        fn += rpad
        bar = '*' * int(round(fi / point_size))
        fmt += f'{fn} {bar} {fi:.1f}\n'
    return fmt


def _de_novo_pred(X, y, feature_names, classifier='decision_tree'):
    if classifier == 'random_forest':
        log.info('Using extremely random forest')
        clf = ExtraTreesClassifier(n_estimators=250, bootstrap=True, oob_score=True)
        clf.fit(X, y)
        log.debug('Feature importance:')
        log.debug(format_feature_importances(feature_names, clf.feature_importances_))
        pred = clf.oob_decision_function_[:, 1]
        # in the unlikely event dt1_pred contains NaNs
        # (can happen when n_estimators is not big enough)
        pred[np.isnan(pred)] = 0
        pred = pred >= 0.5

    else:
        clf = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=100,
            min_impurity_decrease=0.005,
        )
        clf.fit(X, y)
        log.debug('Tree structure:')
        log.debug(export_text(clf, feature_names=feature_names))
        pred = clf.predict(X)
    return pred.astype(int)


def dt1_pred(intron_motif, jad_labels, is_primary_donor, is_primary_acceptor,
             motif_regex='GTAG|GCAG|ATAG', jad_size_threshold=4):
    motif_regex = re.compile(motif_regex)
    is_canon = np.asarray([bool(motif_regex.match(m)) for m in intron_motif])

    jad_labels = np.asarray(jad_labels) >= jad_size_threshold

    is_primary_donor = np.asarray(is_primary_donor, dtype=bool)
    is_primary_acceptor = np.asarray(is_primary_acceptor, dtype=bool)

    is_primary = is_primary_donor & is_primary_acceptor
    return (jad_label & is_canon) | (is_primary & is_canon)


def dt1_de_novo_pred(intron_motif, intron_lengths,
                     jad_labels, is_primary_donor, is_primary_acceptor,
                     is_annot, motif_regex='GTAG|GCAG|ATAG',
                     classifier='decision_tree'):
    motif_regex = re.compile(motif_regex)
    is_canon = np.asarray([int(bool(motif_regex.match(m))) for m in intron_motif])

    jad_labels = np.asarray(jad_labels)

    is_primary_donor = np.asarray(is_primary_donor)
    is_primary_acceptor = np.asarray(is_primary_acceptor)

    intron_length_quantile = quantile_transform(
        np.asarray(intron_lengths).reshape(-1, 1)
    ).ravel()

    X = np.stack(
        [
            is_canon, jad_labels,
            is_primary_donor, is_primary_acceptor,
            intron_length_quantile
        ],
        axis=1
    )
    y = np.asarray(is_annot, dtype=np.int)
    pred = _de_novo_pred(X, y, DT1_DENOVO_FEATURES, classifier=classifier)
    return pred


def dt2_pred(jad_labels,
             is_primary_donor,
             is_primary_acceptor,
             donor_lr_score,
             acceptor_lr_score,
             low_conf_thresh=0.1,
             high_conf_thresh=0.6,
             jad_size_threshold=4):

    jad_labels = np.asarray(jad_labels) >= jad_size_threshold
    is_primary_donor = np.asarray(is_primary_donor, dtype=bool)
    is_primary_acceptor = np.asarray(is_primary_acceptor, dtype=bool)
    donor_lr_score = np.asarray(donor_lr_score, dtype=np.float64)
    acceptor_lr_score = np.asarray(acceptor_lr_score, dtype=np.float64)

    is_primary = is_primary_donor & is_primary_acceptor

    seq_low_conf = ((donor_lr_score >= low_conf_thresh) &
                    (acceptor_lr_score >= low_conf_thresh))
    seq_high_conf = ((donor_lr_score >= high_conf_thresh) &
                     (acceptor_lr_score >= high_conf_thresh))

    return (jad_labels & seq_low_conf) | (is_primary & seq_high_conf)


def dt2_de_novo_pred(intron_lengths, jad_labels,
                     is_primary_donor, is_primary_acceptor,
                     donor_lr_score, acceptor_lr_score,
                     is_annot, classifier='decision_tree'):
    jad_labels = np.asarray(jad_labels)

    is_primary_donor = np.asarray(is_primary_donor)
    is_primary_acceptor = np.asarray(is_primary_acceptor)

    intron_length_quantile = quantile_transform(
        np.asarray(intron_lengths).reshape(-1, 1)
    ).ravel()

    X = np.stack(
        [
            jad_labels, is_primary_donor, is_primary_acceptor,
            intron_length_quantile, donor_lr_score, acceptor_lr_score,
        ],
        axis=1
    )
    y = np.asarray(is_annot, dtype=np.int)
    pred = _de_novo_pred(X, y, DT2_DENOVO_FEATURES, classifier=classifier)
    return pred
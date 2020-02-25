import re
import numpy as np


def dt1_pred(intron_motif, jad_label, is_primary_donor, is_primary_acceptor,
             motif_regex='GTAG|GCAG|ATAG', jad_size_threshold=4):
    motif_regex = re.compile(motif_regex)
    is_canon = np.asarray([bool(motif_regex.match(m)) for m in intron_motif])

    jad_label = np.asarray(jad_label) >= jad_size_threshold

    is_primary_donor = np.asarray(is_primary_donor, dtype=bool)
    is_primary_acceptor = np.asarray(is_primary_acceptor, dtype=bool)

    is_primary = is_primary_donor & is_primary_acceptor
    return (jad_label & is_canon) | (is_primary & is_canon)


def dt2_pred(jad_label,
             is_primary_donor,
             is_primary_acceptor,
             donor_lr_score,
             acceptor_lr_score,
             low_conf_thresh=0.1,
             high_conf_thresh=0.6,
             jad_size_threshold=4):

    jad_label = np.asarray(jad_label) >= jad_size_threshold
    is_primary_donor = np.asarray(is_primary_donor, dtype=bool)
    is_primary_acceptor = np.asarray(is_primary_acceptor, dtype=bool)
    donor_lr_score = np.asarray(donor_lr_score, dtype=np.float64)
    acceptor_lr_score = np.asarray(acceptor_lr_score, dtype=np.float64)

    is_primary = is_primary_donor & is_primary_acceptor

    seq_low_conf = ((donor_lr_score >= low_conf_thresh) &
                    (acceptor_lr_score >= low_conf_thresh))
    seq_high_conf = ((donor_lr_score >= high_conf_thresh) &
                     (acceptor_lr_score >= high_conf_thresh))

    return (jad_label & seq_low_conf) | (is_primary & seq_high_conf)
from collections import defaultdict

import numpy as np

import pysam
from joblib import Parallel, delayed


RC = str.maketrans('ACGT', 'TGCA')


def rev_comp(seq):
    return seq.translate(RC)[::-1]


def fetch_padded(fasta, chrom, pos, w):
    clen = fasta.get_reference_length(chrom)
    left = pos - w
    right = pos + w
    if left < 0:
        lpad = abs(left)
        left = 0
    else:
        lpad = 0
    if right > clen:
        rpad = right - clen
        right = clen
    else:
        rpad = 0
    seq = fasta.fetch(chrom, left, right)
    if lpad:
        seq = 'N' * lpad + seq
    if rpad:
        seq = seq + 'N' * rpad
    return seq
    

def _get_junc_seqs(bed_records, fasta_fn, window_size):
    intron_donor_labels = defaultdict(lambda: 0)
    intron_acceptor_labels = defaultdict(lambda: 0)
    intron_donor_seqs = {}
    intron_acceptor_seqs = {}
    w = window_size // 2
    with pysam.FastaFile(fasta_fn) as fasta:
        for chrom, start, end, strand, label in bed_records:
            donor_seq = fetch_padded(fasta, chrom, start, w)
            acceptor_seq = fetch_padded(fasta, chrom, end, w)
            if strand == '-':
                donor_seq, acceptor_seq = (
                    rev_comp(acceptor_seq), rev_comp(donor_seq)
                )
                donor_pos = end
                acceptor_pos = start
            else:
                donor_pos = start
                acceptor_pos = end

            intron_donor_seqs[(chrom, donor_pos, strand)] = donor_seq
            intron_donor_labels[(chrom, donor_pos, strand)] |= label
            intron_acceptor_seqs[(chrom, acceptor_pos, strand)] = acceptor_seq
            intron_acceptor_labels[(chrom, acceptor_pos, strand)] |= label

    return (intron_donor_seqs, intron_donor_labels,
            intron_acceptor_seqs, intron_acceptor_labels)


def chunk_records(introns, labels, processes):
    records = []
    for (chrom, start, end, strand), lab in zip(introns, labels):
        records.append((chrom, start, end, strand, lab))
    nrecords = len(records)
    n, r = divmod(nrecords, processes)
    split_points = ([0] + r * [n + 1] + (processes - r) * [n])
    split_points = np.cumsum(split_points)
    for i in range(processes):
        start = split_points[i]
        end = split_points[i + 1]
        yield records[start: end]


def or_update(d1, d2):
    for k, v in d2.items():
        d1[k] |= v
    return d1


def merge_parallel_junc_res(res):
    donor_seqs = {}
    donor_labels = defaultdict(lambda: 0)
    acceptor_seqs = {}
    acceptor_labels = defaultdict(lambda: 0)

    for ds, dl, as_, al in res:
        donor_seqs.update(ds)
        acceptor_seqs.update(as_)
        donor_labels = or_update(donor_labels, dl)
        acceptor_labels = or_update(acceptor_labels, al)
    
    donors = list(donor_seqs.keys())
    donor_seqs = [donor_seqs[d] for d in donors]
    donor_labels = [donor_labels[d] for d in donors]

    acceptors = list(acceptor_seqs.keys())
    acceptor_seqs = [acceptor_seqs[a] for a in acceptors]
    acceptor_labels = [acceptor_labels[a] for a in acceptors]

    return (donors, donor_seqs, donor_labels,
            acceptors, acceptor_seqs, acceptor_labels)


def get_junction_seqs(introns, labels, fasta_fn, window_size, processes=12):
    with Parallel(n_jobs=processes) as pool:
        res = pool(
            delayed(_get_junc_seqs)(introns, fasta_fn, window_size)
            for introns in chunk_records(introns, labels, processes)
        )

    (donors, donor_seqs, donor_labels,
     acceptors, acceptor_seqs, acceptor_labels) = merge_parallel_junc_res(res)

    return (donors, donor_seqs, donor_labels,
            acceptors, acceptor_seqs, acceptor_labels)
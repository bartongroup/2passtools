from collections import defaultdict, Counter
import re

import numpy as np
import pysam
from joblib import Parallel, delayed
from ncls import NCLS


CS_SPLITTER = '([-+*~=:])'
RC = str.maketrans('ACGTN', 'TGCAN')

def parse_cs_tag(cs_tag):
    '''
    generalisable function for parsing minimap2 cs tag (long and short form)
    '''
    cs_tag = re.split(CS_SPLITTER, cs_tag)[1:]
    cs_ops = cs_tag[::2]
    cs_info = cs_tag[1::2]
    cs_parsed = []
    for op, info in zip(cs_ops, cs_info):
        if op == '=':
            # long form match
            cs_parsed.append(('=', len(info), info))
        elif op == ':':
            # short form match
            cs_parsed.append(('=', int(info), None))
        elif op == '*':
            # mismatch
            ref = info[0]
            alt = info[1]
            cs_parsed.append((op, 1, (ref, alt)))
        elif op == '+' or op == '-':
            cs_parsed.append((op, len(info), info))
        elif op == '~':
            donor_motif, intron_length, acceptor_motif = re.match(
                '^([acgtn]{2})([0-9]+)([acgtn]{2})', info).groups()
            motif = (donor_motif + acceptor_motif).upper()
            intron_length = int(intron_length)
            cs_parsed.append((op, intron_length, motif))
    return cs_parsed


def get_junction_overhang_size(overhang_cs):
    '''
    for cs tag split at intron (and reoriented so nearest op to intron is first)
    returns the overhang size (number of nt which match before
    first mismatch, insertion or deletion)
    '''
    try:
        return overhang_cs[0][1] if overhang_cs[0][0] == '=' else 0
    except IndexError:
        # sometimes when junctions are provided minimap2 can produce alignments
        # where an annotated junction is used with no overhang on the other
        # side!!
        return 0


def infer_strand_from_intron_motifs(intron_motifs, read_strand):
    strand_counts = Counter()
    for motif in intron_motifs:
        if re.match('G[TC]AG', motif):
            strand_counts['+'] += 1
        elif re.match('CT[AG]C', motif):
            strand_counts['-'] += 1
        else:
            strand_counts['.'] += 1

    if strand_counts['+'] == strand_counts['-']:
        return read_strand
    elif strand_counts['+'] > strand_counts['-']:
        return '+'
    else:
        return '-'
        

def find_introns(aln, stranded=True):
    '''
    use the cs tag to find introns and their match overhangs in the alignment
    '''
    introns = []
    intron_motifs = []
    chrom = aln.reference_name
    start = aln.reference_start
    end = aln.reference_end
    read_strand = '+-'[aln.is_reverse]
    pos = start
    cs_tag = parse_cs_tag(aln.get_tag('cs'))
    for i, (op, ln, info) in enumerate(cs_tag):
        if op == '+':
            # insertion does not consume reference
            continue
        elif op in ('=', '*', '-'):
            # match, mismatch, deletion consume reference
            pos += ln
        elif op == '~':
            # intron consumes reference and is recorded
            left = pos
            right = left + ln
            left_tag = cs_tag[:i][::-1]
            right_tag = cs_tag[i + 1:]
            junc_overhang = min(
                get_junction_overhang_size(left_tag),
                get_junction_overhang_size(right_tag)
            )
            # info is intron motif
            introns.append([left, right, junc_overhang, ln, info])
            intron_motifs.append(info)
            pos = right

    # infer strand and yield introns
    if stranded:
        strand = read_strand
    else:
        strand = infer_strand_from_intron_motifs(intron_motifs, read_strand)

    n_introns = len(introns)
    for i, (start, end, overhang, length, motif) in enumerate(introns, 1):
        if strand == '-':
            motif = motif.translate(RC)[::-1]
        yield chrom, start, end, strand, motif, overhang, length


def build_donor_acceptor_ncls(introns, intron_counts, intron_jads, dist=20):
    donor_invs = defaultdict(Counter)
    acceptor_invs = defaultdict(Counter)
    donor_inv_jads = defaultdict(Counter)
    acceptor_inv_jads = defaultdict(Counter)

    for (chrom, start, end, strand), count, jad in zip(introns, intron_counts, intron_jads):
        if strand == '+':
            donor_inv = (start - dist, start + dist, start)
            acceptor_inv = (end - dist, end + dist, end)
        else:
            donor_inv = (end - dist, end + dist, end)
            acceptor_inv = (start - dist, start + dist, start)
        donor_invs[(chrom, strand)][donor_inv] += count
        acceptor_invs[(chrom, strand)][acceptor_inv] += count

        # jad is used to break count ties
        donor_inv_jads[(chrom, strand)][donor_inv] = max(
            donor_inv_jads[(chrom, strand)][donor_inv], jad
        )
        acceptor_inv_jads[(chrom, strand)][acceptor_inv] = max(
            acceptor_inv_jads[(chrom, strand)][acceptor_inv], jad
        )

    da_itree = {}
    for label, invs, inv_jads in zip(['donor', 'acceptor'],
                                     [donor_invs, acceptor_invs],
                                     [donor_inv_jads, acceptor_inv_jads]):
        da_itree[label] = {}
        for chrom, pos in invs.items():
            jads = [inv_jads[chrom][i] for i in pos]
            starts, ends, mids, counts = zip(*[(s, e, m, c) for (s, e, m), c in pos.items()])
            starts = np.array(starts, dtype=np.int64)
            ends = np.array(ends, dtype=np.int64)
            idx = np.array(mids, dtype=np.int64)
            counts = {i: (c, j) for i, c, j in zip(mids, counts, jads)}
            itree = NCLS(starts, ends, idx)
            da_itree[label][chrom] = (itree, counts)
    return da_itree


def assign_primary(chrom, start, end, strand, inv_trees):
    donor_pos = start if strand == '+' else end
    acceptor_pos = end if strand == '+' else start

    is_primary = {}
    for label, pos, in zip(['donor', 'acceptor'],
                           [donor_pos, acceptor_pos]):
        itree, counts = inv_trees[label][(chrom, strand)]
        max_count = 0
        max_jad = 0
        for _, _, ov_pos in itree.find_overlap(pos, pos):
            if ov_pos != pos:
                c, j = counts[ov_pos]
                max_count = max(max_count, c)
                max_jad = max(max_jad, j)
        if max_count < counts[pos][0]:
            is_primary[label] = True
        elif (max_count == counts[pos][0]) & (max_jad < counts[pos][1]):
            # break count ties with jad
            is_primary[label] = True
        else:
            is_primary[label] = False
    return is_primary['donor'], is_primary['acceptor']


def fetch_introns_for_interval(bam_fn, chrom, start, end, stranded):
    motifs = {}
    lengths = {}
    counts = Counter()
    intron_jads = Counter()
    with pysam.AlignmentFile(bam_fn) as bam:
        for aln in bam.fetch(chrom, start, end):
            # to prevent double counting of introns, ignore alns
            # which start before beginning of specified interval
            if aln.reference_start < start:
                continue
            for *i, m, ov, ln in find_introns(aln, stranded):
                i = tuple(i)
                motifs[i] = m
                lengths[i] = ln
                counts[i] += 1
                intron_jads[i] = max(intron_jads[i], ov)
    return motifs, lengths, counts, intron_jads


def get_bam_intervals(bam_fn, batch_size):
    with pysam.AlignmentFile(bam_fn) as bam:
        references = {ref: ln for ref, ln in zip(bam.references, bam.lengths)}
    for ref, ref_len in references.items():
        for i in range(0, ref_len, batch_size):
            query = (ref, i, min(ref_len, i + batch_size))
            yield query


def merge_intron_res(res):
    motifs = {}
    lengths = {}
    counts = Counter()
    intron_jads = Counter()
    for m, l, c, j in res:
        motifs.update(m)
        lengths.update(l)
        counts += c
        for i, jad in j.items():
            intron_jads[i] = max(intron_jads[i], jad)
    return motifs, lengths, counts, intron_jads


def parse_introns(bam_fn, primary_splice_local_dist,
                  stranded, batch_size, processes):
    '''
    find all introns in the dataset, label them as positive or negative
    training examples using the simple jad filter and then extract their
    sequences from the reference for training the neural network
    '''
    with Parallel(n_jobs=processes) as pool:
        res = pool(
            delayed(fetch_introns_for_interval)(
                    bam_fn, *inv, stranded)
            for inv in get_bam_intervals(bam_fn, batch_size)
        )
    motifs, lengths, counts, intron_jads = merge_intron_res(res)

    introns = list(motifs.keys())
    motifs = [motifs[i] for i in introns]
    lengths = [lengths[i] for i in introns]
    counts = [counts[i] for i in introns]
    jad_label = [intron_jads[i] for i in introns]

    itrees = build_donor_acceptor_ncls(
        introns, counts, jad_label, primary_splice_local_dist
    )
    is_primary_donor = []
    is_primary_acceptor = []
    for i in introns:
        d, a = assign_primary(*i, itrees)
        is_primary_donor.append(d)
        is_primary_acceptor.append(a)

    return (introns, motifs, lengths, counts, jad_label,
            is_primary_donor, is_primary_acceptor)

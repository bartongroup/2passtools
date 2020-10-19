from .bamparse import merge_intron_res, build_donor_acceptor_ncls, assign_primary


def read_junc_bed(bed_fn):
    motifs = {}
    lengths = {}
    counts = {}
    intron_jads = {}
    with open(bed_fn) as bed:
        for record in bed:
            (chrom, start, end, motif, count, strand,
             jad, *_) = record.split()
            start = int(start)
            end = int(end)
            ln = end - start
            count = int(count)
            jad = int(jad)
            term = int(term)
            i = (chrom, start, end, strand)
            motifs[i] = motif
            lengths[i] = ln
            counts[i] = count
            intron_jads[i] = jad
    return motifs, lengths, counts, intron_jads


def get_merged_juncs(junc_bed_fns, primary_splice_local_dist=20):

    res = [read_junc_bed(fn) for fn in junc_bed_fns]
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
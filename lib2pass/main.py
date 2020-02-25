import os
import logging

import click
import click_log

import numpy as np
from .bamparse import parse_introns
from .seqlr import predict_splice_junctions_from_seq
from .decisiontree import dt1_pred, dt2_pred
from .merge import get_merged_juncs
from .filter import apply_eval_expression

log = logging.getLogger('2passtools')
click_log.basic_config(log)


def _all_predictions(introns, motifs, counts, jad_labels,
                     is_primary_donor, is_primary_acceptor,
                     ref_fasta_fn,
                     canonical_motifs, jad_size_threshold,
                     lr_window_size, lr_kfold,
                     lr_low_confidence_threshold,
                     lr_high_confidence_threshold,
                     processes):
    log.info(f'Identified {len(introns):d} introns')
    dt1_labels = dt1_pred(
        motifs, jad_labels, is_primary_donor, is_primary_acceptor,
        motif_regex=canonical_motifs,
        jad_size_threshold=jad_size_threshold,
    )
    log.info(f'{sum(dt1_labels):d} introns pass filter dt1')
    lr_donor_labels, lr_acceptor_labels = predict_splice_junctions_from_seq(
        introns, dt1_labels, ref_fasta_fn,
        lr_window_size, lr_kfold,
        processes
    )
    dt2_labels = dt2_pred(
        jad_labels, is_primary_donor, is_primary_acceptor,
        lr_donor_labels, lr_acceptor_labels,
        lr_low_confidence_threshold, lr_high_confidence_threshold,
        jad_size_threshold=jad_size_threshold
    )
    log.info(f'{sum(dt2_labels):d} introns pass filter dt2')
    return (
        introns, motifs, counts, jad_labels,
        is_primary_donor, is_primary_acceptor,
        dt1_labels,
        lr_donor_labels, lr_acceptor_labels,
        dt2_labels
    )


def validate_motif_regex(ctx, param, value):
    if not set(value).issubset(set('ACGT|')):
        raise click.BadParameter('unrecognised motifs, use only ACGT and | to separate')
    for m in value.split('|'):
        if not len(m) == 4:
            raise click.BadParameter('all motifs should be 4 nt')
    else:
        return value


@click.group()
def main():
    pass


@main.command()
@click.argument('bam-fn', required=True, nargs=1)
@click.option('-o', '--output-bed-fn', required=True)
@click.option('-f', '--ref-fasta-fn', required=True, type=str)
@click.option('-j', '--jad-size-threshold', default=4)
@click.option('-d', '--primary-splice-local-dist', default=20)
@click.option('-m', '--canonical-motifs', default='GTAG|GCAG|ATAG', callback=validate_motif_regex)
@click.option('-w', '--lr-window-size', default=128, type=int)
@click.option('-k', '--lr-kfold', default=6, type=int)
@click.option('-lt', '--lr-low-confidence-threshold', default=0.1, type=float)
@click.option('-ht', '--lr-high-confidence-threshold', default=0.6, type=float)
@click.option('--stranded/--unstranded', default=True)
@click.option('-p', '--processes', default=1)
@click.option('-s', '--random-seed', default=None, type=int)
@click_log.simple_verbosity_option(log)
def score(bam_fn, output_bed_fn, ref_fasta_fn,
          jad_size_threshold,
          primary_splice_local_dist, canonical_motifs,
          lr_window_size, lr_kfold,
          lr_low_confidence_threshold, lr_high_confidence_threshold,
          stranded, processes, random_seed):

    if random_seed is not None:
        np.random.seed(random_seed)

    log.info(f'Parsing BAM file: {bam_fn}')
    (introns, motifs, counts, jad_labels,
     is_primary_donor, is_primary_acceptor) = parse_introns(
        bam_fn,
        primary_splice_local_dist,
        stranded,
        1_000_000, processes
    )
    res = zip(*_all_predictions(
        introns, motifs, counts, jad_labels,
        is_primary_donor, is_primary_acceptor,
        ref_fasta_fn,
        canonical_motifs, jad_size_threshold,
        lr_window_size, lr_kfold,
        lr_low_confidence_threshold,
        lr_high_confidence_threshold,
        processes
    ))
    log.info(f'Writing results to {output_bed_fn}')
    with open(output_bed_fn, 'w') as bed:
        for i, motif, c, jad, pd, pa, d1, lrd, lra, d2 in res:
            chrom, start, end, strand = i
            bed.write(
                f'{chrom:s}\t{start:d}\t{end:d}\t{motif:s}\t{c:d}\t{strand:s}\t'
                f'{jad:d}\t{pd:d}\t{pa:d}\t{d1:d}\t'
                f'{lrd:.3f}\t{lra:.3f}\t{d2:d}\n'
            )


@main.command()
@click_log.simple_verbosity_option(log)
@click.argument('bed-fns', nargs=-1, required=True)
@click.option('-o', '--output-bed-fn', required=True)
@click.option('-f', '--ref-fasta-fn', required=True, type=str)
@click.option('-j', '--jad-size-threshold', default=4)
@click.option('-d', '--primary-splice-local-dist', default=20)
@click.option('-m', '--canonical-motifs', default='GTAG|GCAG|ATAG', callback=validate_motif_regex)
@click.option('-w', '--lr-window-size', default=128, type=int)
@click.option('-k', '--lr-kfold', default=6, type=int)
@click.option('-lt', '--lr-low-confidence-threshold', default=0.1, type=float)
@click.option('-ht', '--lr-high-confidence-threshold', default=0.6, type=float)
@click.option('-p', '--processes', default=1)
@click.option('-s', '--random-seed', default=None, type=int)
def merge(bed_fns, output_bed_fn, ref_fasta_fn,
          jad_size_threshold, primary_splice_local_dist, canonical_motifs,
          lr_window_size, lr_kfold,
          lr_low_confidence_threshold, lr_high_confidence_threshold,
          processes, random_seed):

    if random_seed is not None:
        np.random.seed(random_seed)

    log.info(f'Parsing {len(bed_fns):d} BED files')
    (introns, motifs, counts, jad_labels,
     is_primary_donor, is_primary_acceptor) = get_merged_juncs(
        bed_fns, primary_splice_local_dist
    )
    res = zip(*_all_predictions(
        introns, motifs, counts, jad_labels,
        is_primary_donor, is_primary_acceptor,
        ref_fasta_fn,
        canonical_motifs, jad_size_threshold,
        lr_window_size, lr_kfold,
        lr_low_confidence_threshold,
        lr_high_confidence_threshold,
        processes
    ))
    log.info(f'Writing results to {output_bed_fn}')
    with open(output_bed_fn, 'w') as bed:
        for i, motif, c, jad, pd, pa, d1, lrd, lra, d2 in res:
            chrom, start, end, strand = i
            bed.write(
                f'{chrom:s}\t{start:d}\t{end:d}\t{motif:s}\t{c:d}\t{strand:s}\t'
                f'{jad:d}\t{pd:d}\t{pa:d}\t{d1:d}\t'
                f'{lrd:.3f}\t{lra:.3f}\t{d2:d}\n'
            )


@main.command()
@click.argument('bed-fn', nargs=1, required=True)
@click.option('-o', '--output-bed-fn', required=True)    
@click.option('--exprs', required=False, default="decision_tree_2_pred")
@click_log.simple_verbosity_option(log)
def filter(bed_fn, output_bed_fn, exprs):
    with open(output_bed_fn, 'w') as bed:
        for chrom, start, end, strand, decision in apply_eval_expression(bed_fn, exprs):
            if decision:
                record = f'{chrom}\t{start}\t{end}\tintron\t0\t{strand}\n'
                bed.write(record)


@main.command()
@click_log.simple_verbosity_option(log)
def mm2pass():
    raise NotImplementedError('TODO: implement convience tool to wrap '
                              'minimap2 and run two pass alignment')


if __name__ == '__main__':
    main()
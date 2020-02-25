'''
lib2pass.main: contains the command line interface for 2passtools
'''
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
    '''
    Takes as input the alignment metrics extracted either from a bam file (2passtools score)
    or previously created junction bed file (2passtools merge). Calculates decision tree
    score one, then extracts junction sequences from the fasta file and calculates
    decision tree score two.
    '''
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


SCORE_MERGE_COMMON_OPTIONS = [
    click.option('-o', '--output-bed-fn', required=True, help='Output file path'),
    click.option('-f', '--ref-fasta-fn', required=True, type=str,
                  help='Path to the fasta file that reads were mapped to'),
    click.option('-j', '--jad-size-threshold', default=4, help='JAD to threshold at in the decision tree'),
    click.option('-d', '--primary-splice-local-dist', default=20,
                  help='Distance to search for alternative donor/acceptors when calculating primary d/a'),
    click.option('-m', '--canonical-motifs', default='GTAG|GCAG|ATAG', callback=validate_motif_regex,
                  help=('Intron motifs considered canonical in organism. '
                        'Should be four char DNA motifs separated by vertical bar only')),
    click.option('-w', '--lr-window-size', default=128, type=int,
                  help='Sequence size to extract to train logistic regression models'),
    click.option('-k', '--lr-kfold', default=6, type=int,
                  help='Number of cross validation k-folds for logistic regression models'),
    click.option('-lt', '--lr-low-confidence-threshold', default=0.1, type=float,
                  help='Logistic regression low confidence threshold for decision tree 2'),
    click.option('-ht', '--lr-high-confidence-threshold', default=0.6, type=float,
                  help='Logistic regression high confidence threshold for decision tree 2'),
    click.option('--stranded/--unstranded', default=True,
                  help=('Whether input data is stranded or unstranded. '
                        'direct RNA is stranded, cDNA often isn\'t')),
    click.option('-p', '--processes', default=1),
    click.option('-s', '--random-seed', default=None, type=int),
    click_log.simple_verbosity_option(log),
]

def _common_options(common_options):
    def _apply_common_options(func):
        for option in reversed(common_options):
            func = option(func)
        return func
    return _apply_common_options


@main.command()
@click.argument('bam-fn', required=True, nargs=1)
@_common_options(SCORE_MERGE_COMMON_OPTIONS)
def score(bam_fn, output_bed_fn, ref_fasta_fn,
          jad_size_threshold,
          primary_splice_local_dist, canonical_motifs,
          lr_window_size, lr_kfold,
          lr_low_confidence_threshold, lr_high_confidence_threshold,
          stranded, processes, random_seed):
    '''
    2passtools score: A tool for extracting and scores junctions from a bam file
    aligned with minimap2. Filtered junctions can be used to realign reads in
    a second pass with minimap2.

    Bam files must be mapped with minimap2 and have the long form CS tag, e.g.

    minimap2 -a --cs=long -k14 -x splice ref.fa reads.fq
    '''

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
@_common_options(SCORE_MERGE_COMMON_OPTIONS)
def merge(bed_fns, output_bed_fn, ref_fasta_fn,
          jad_size_threshold, primary_splice_local_dist, canonical_motifs,
          lr_window_size, lr_kfold,
          lr_low_confidence_threshold, lr_high_confidence_threshold,
          processes, random_seed):
    '''
    2passtools merge: Merges bed files produced by 2passtools score on individual
    replicates and recalculates junction strength metrics. Can be used to create
    a unified junction set to realign reads from different replicates.
    '''
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
    '''
    2passtools filter: Convenience tool to filter a junction bed and produce
    6-column bed format which is compatible with minimap2.
    '''
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
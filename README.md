# 2passtools

[![doi](https://zenodo.org/badge/242980365.svg)](https://doi.org/10.5281/zenodo.3778819)

A package for filtering splice junctions extracted from noisy long read alignments generated using minimap2. These can then be used to perform second pass alignment with minimap2, feeding in the junctions using the `--junc-bed` flag.

## Installation:

2passtools has been tested with python 3.6, and requires `numpy`, `scikit-learn`, `pysam`, `NCLS` and `click`. The easiest way to install it is using the conda environment yaml provided:

```

git clone https://www.github.com/bartongroup/2passtools
cd 2passtools
conda env create -f 2passtools.yaml

source activate 2passtools
```

Alternatively 2passtools and the required packages can be installed using pip:

```
pip install git+git://github.com/bartongroup/2passtools.git
```


## Use:

2passtools has three commands....

NB: There is a [snakemake](https://www.github.com/bartongroup/two_pass_alignment_pipeline) pipeline which can be used to run the benchmarking scripts used in the manuscript.

### `score`:

The `2passtools score` command requires as input a long read sequencing bam file aligned using minimap2 and a reference fasta file. It then extracts junction metrics and sequence information and uses it to score splice junctions found in the alignments. The output of `score` is a BED file with multiple columns corresponding to different metrics and model scores (see output below). This format cannot be passed to minimap2 directly as (A) it has not yet been filtered and (B) the extra column format is not supported by minimap2 which requires 6-column bed. Filtering and reformatting can be done using `2passtools filter`.

#### Options:
 
```
$ 2passtools score --help
Usage: 2passtools score [OPTIONS] BAM_FN

  2passtools score: A tool for extracting and scores junctions from a bam
  file aligned with minimap2. Filtered junctions can be used to realign
  reads in a second pass with minimap2.

  Bam file must be mapped with minimap2 and have the long form CS tag, e.g.

  minimap2 -a --cs=long -k14 -x splice ref.fa reads.fq

Options:
  -o, --output-bed-fn TEXT        Output file path  [required]
  -f, --ref-fasta-fn TEXT         Path to the fasta file that reads were
                                  mapped to  [required]
  -j, --jad-size-threshold INTEGER
                                  JAD to threshold at in the decision tree
  -d, --primary-splice-local-dist INTEGER
                                  Distance to search for alternative
                                  donor/acceptors when calculating primary d/a
  -m, --canonical-motifs TEXT     Intron motifs considered canonical in
                                  organism. Should be four char DNA motifs
                                  separated by vertical bar only
  -w, --lr-window-size INTEGER    Sequence size to extract to train logistic
                                  regression models
  -k, --lr-kfold INTEGER          Number of cross validation k-folds for
                                  logistic regression models
  -lt, --lr-low-confidence-threshold FLOAT
                                  Logistic regression low confidence threshold
                                  for decision tree 2
  -ht, --lr-high-confidence-threshold FLOAT
                                  Logistic regression high confidence
                                  threshold for decision tree 2
  --stranded / --unstranded       Whether input data is stranded or
                                  unstranded. direct RNA is stranded, cDNA
                                  often isn't
  -p, --processes INTEGER
  -s, --random-seed INTEGER
  -v, --verbosity LVL             Either CRITICAL, ERROR, WARNING, INFO or
                                  DEBUG
  --help                          Show this message and exit.

```

#### Output:

A 13-column BED file format with the following values:

```
1. chrom (string)
2. start (integer)
3. end (integer)
4. intron-motif (four char string)
5. supporting read count (integer)
6. strand (string, either '+' or '-')
7. junction alignment distance metric (integer)
8. primary donor metric (integer, either 0 or 1)
9. primary acceptor metric (integer, either 0 or 1)
10. decision tree 1 output (integer, either 0 or 1)
11. logistic regression model donor score (float)
12. logistic regression model acceptor score (float)
13. decision tree 2 output (integer, either 0 or 1)
```

### `filter`:

The `2passtools filter` command can be used to filter the 13-column bed file using any expression utilising the metrics or model outputs. The expression should be a valid python expression which evaluates to `True` or `False` for each junction, and can use any of the following safe variables and functions:

* `motif`: The intron motif in ACGTN alphabet (`str`),
* `is_GTAG`: The intron motif is GU/AG (`bool`),
* `is_GCAG`: The intron motif is GC/AG (`bool`),
* `is_ATAG`: The intron motif is AU/AG (`bool`),
* `motif_regex_match`: safe function allowing regex matching of motif, e.g. `motif_regex_match("G[CT]AG")` (`func`),
* `count`: The supporting read count (`int`),
* `jad`: The junction alignment distance metric (`int`),
* `primary_donor`: The primary donor metric (`bool`),
* `primary_acceptor`: The primary acceptor metric (`bool`),
* `donor_seq_score`: The logistic regression model donor score (`float`),
* `acceptor_seq_score`: The logistic regression model acceptor score (`float`),
* `decision_tree_1_pred`: Decision tree model 1 output (`bool`),
* `decision_tree_2_pred`: Decision tree model 2 output (`bool`),
* `sum`, `pow`, `min`, `max`, `len`: python functions,
* `math`: The python `math` module, any function from it is useable,
* `bool`, `int`, `str`, `float`: python functions.

For example:

* `2passtools filter --exprs 'jad > 3'` filters for junction alignment distance of 4 nt or more.
* `2passtools filter --exprs 'decision_tree_2_pred'` filters for junctions that pass the second decision tree model.

etc.

#### Usage:

```
$ 2passtools filter --help
Usage: 2passtools filter [OPTIONS] BED_FN

  2passtools filter: Convenience tool to filter a junction bed and produce
  6-column bed format which is compatible with minimap2.

Options:
  -o, --output-bed-fn TEXT  [required]
  --exprs TEXT
  -v, --verbosity LVL       Either CRITICAL, ERROR, WARNING, INFO or DEBUG
  --help                    Show this message and exit.
```

### `merge`:

The `2passtools merge` command is similar to `score`, but takes multiple 13-column bed files produced by `score` and merges them, recalculating metrics and model stats, to produce a unified junction set. This is useful for making sure all replicates are aligned similarly, and often alignment is improved by borrowing power across replicates. Output is in the same 13-column BED format as `score`.

#### `Usage`:

```
$ 2passtools filter --help
Usage: 2passtools filter [OPTIONS] BED_FN

  2passtools filter: Convenience tool to filter a junction bed and produce
  6-column bed format which is compatible with minimap2.

Options:
  -o, --output-bed-fn TEXT  [required]
  --exprs TEXT
  -v, --verbosity LVL       Either CRITICAL, ERROR, WARNING, INFO or DEBUG
  --help                    Show this message and exit.
(4006b908) [mtparker@ningal envs]$ 2passtools merge --help
Usage: 2passtools merge [OPTIONS] BED_FNS...

  2passtools merge: Merges bed files produced by 2passtools score on
  individual replicates and recalculates junction strength metrics. Can be
  used to create a unified junction set to realign reads from different
  replicates.

  Bed files should be in the 13 column format produced by 2passtools score.

Options:
  -o, --output-bed-fn TEXT        Output file path  [required]
  -f, --ref-fasta-fn TEXT         Path to the fasta file that reads were
                                  mapped to  [required]
  -j, --jad-size-threshold INTEGER
                                  JAD to threshold at in the decision tree
  -d, --primary-splice-local-dist INTEGER
                                  Distance to search for alternative
                                  donor/acceptors when calculating primary d/a
  -m, --canonical-motifs TEXT     Intron motifs considered canonical in
                                  organism. Should be four char DNA motifs
                                  separated by vertical bar only
  -w, --lr-window-size INTEGER    Sequence size to extract to train logistic
                                  regression models
  -k, --lr-kfold INTEGER          Number of cross validation k-folds for
                                  logistic regression models
  -lt, --lr-low-confidence-threshold FLOAT
                                  Logistic regression low confidence threshold
                                  for decision tree 2
  -ht, --lr-high-confidence-threshold FLOAT
                                  Logistic regression high confidence
                                  threshold for decision tree 2
  -p, --processes INTEGER
  -s, --random-seed INTEGER
  -v, --verbosity LVL             Either CRITICAL, ERROR, WARNING, INFO or
                                  DEBUG
  --help                          Show this message and exit.
  ```

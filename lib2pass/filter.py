import math
from collections import defaultdict
from functools import partial


def nullable(val, cast):
    if val is None:
        return None
    else:
        return cast(val)


def eval_feature_expression(
        record,
        expression):
    (
        motif, count, jad_score,
        is_primary_donor, is_primary_acceptor,
        dt1_pred,
        donor_seq_score, acceptor_seq_score,
        dt2_pred
    ) = record
    safe_dict = {
        'motif': motif,
        'is_GTAG': motif == 'GTAG',
        'is_GCAG': motif == 'GCAG',
        'is_ATAG': motif == 'ATAG',
        'motif_regex_match': lambda expr: bool(re.match(expr, motif)),
        'count': count,
        'jad': jad_score,
        'primary_donor': bool(is_primary_donor),
        'primary_acceptor': bool(is_primary_acceptor),
        'donor_seq_score': donor_seq_score,
        'acceptor_seq_score': acceptor_seq_score,
        'decision_tree_1_pred': bool(dt1_pred),
        'decision_tree_2_pred': bool(dt2_pred),
        'sum': sum,
        'pow': pow,
        'min': min,
        'max': max,
        'math': math,
        'bool': bool,
        'int': partial(nullable, int),
        'str': partial(nullable, str),
        'float': partial(nullable, float),
        'len': partial(nullable, len),
    }
    res = eval(expression, {"__builtins__": None}, safe_dict)
    if not isinstance(res, bool):
        res = bool(res)
    return res


def read_junc_bed(bed_fn):
    records = {}
    with open(bed_fn) as bed:
        for record in bed:
            (chrom, start, end, motif, count, strand,
             jad, is_pd, is_pa, dt1, lra, lrd, dt2) = record.split()
            records[(chrom, start, end, strand)] = (
                motif, int(count),
                int(jad), int(is_pd), int(is_pa),
                int(dt1), float(lrd), float(lra), int(dt2)
            )
    return records


def apply_eval_expression(bed_fn, expression):
    for (chrom, start, end, strand), scores in read_junc_bed(bed_fn).items():
        r = eval_feature_expression(
            scores, expression
        )
        yield chrom, start, end, strand, r
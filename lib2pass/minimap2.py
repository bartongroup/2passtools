import os
import subprocess
import tempfile

MINIMAP2 = os.path.abspath(
    os.path.split(__file__)[0] + 
    '/../external/minimap2/minimap2'
)


def subprocess_command(cmd, stdout_fn):
    with open(stdout_fn, 'w') as s:
        proc = subprocess.Popen(
            cmd,
            stdout=s,
            stderr=subprocess.PIPE
        )
        _, stderr = proc.communicate()
        if proc.returncode:
            raise subprocess.CalledProcessError(stderr.decode())
        else:
            return stderr.decode()


def map_with_minimap2(fastq_fn, reference_fn, output_fn, threads=1,
                      use_canon=False, noncanon_pen=9,
                      junc_bed=None, junc_bonus=9):
    if not os.path.exists(fastq_fn):
        raise OSError('fastq_fn not found')
    elif not os.path.exists(reference_fn):
        raise OSError('reference_fn not found')
    splice_flank = 'yes' if use_canon else 'no'
    noncanon_pen = noncanon_pen if use_canon else 0
    use_canon = 'f' if use_canon else 'n'
    s_handle, sam_fn = tempfile.mkstemp(suffix='.sam')
    b_handle, bam_fn = tempfile.mkstemp(suffix='.bam')

    # run minimap
    minimap2_cmd = [
        MINIMAP2, f'-t{threads}', '-k14', '-w5', '--splice',
        '-g2000', '-G10000', '-A1', '-B2', '-O2,32', '-E1,0',
        f'-C{noncanon_pen}', f'--splice-flank={splice_flank}', f'-u{use_canon}',
        '-z200', '-L', '--cs=long', '-a'
    ]
    if junc_bed is not None:
        minimap2_cmd += ['--junc-bed', junc_bed, f'--junc-bonus={junc_bonus}']
    minimap2_cmd += [reference_fn, fastq_fn]
    minimap2_stderr = subprocess_command(minimap2_cmd, sam_fn)

    # run samtools view
    samtools_view_cmd = ['samtools', 'view', '-bS', sam_fn]
    samtools_view_stderr = subprocess_command(samtools_view_cmd, bam_fn)

    # clean up minimap2 output
    os.close(s_handle)
    os.remove(sam_fn)

    # run samtools sort
    samtools_sort_cmd = ['samtools', 'sort', '-@', str(threads), '-o', '-', bam_fn]
    samtools_sort_stderr = subprocess_command(samtools_sort_cmd, output_fn)

    # clean up samtools view output
    os.close(b_handle)
    os.remove(bam_fn)

    # run samtools index
    samtools_index_cmd = ['samtools', 'index', output_fn]
    subprocess.check_call(samtools_index_cmd)
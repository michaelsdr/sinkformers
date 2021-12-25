import os
import time


def write_script(sh_name, name, args_string):
    '''
    Writes the bash script to launch expe
    '''
    with open('%s.sh' % sh_name, 'w') as rsh:
        rsh.write('''\
#!/bin/bash
#SBATCH -A ynt@gpu
#SBATCH --job-name=%s%%j     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1               # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=16:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=job_outputs/%s%%j.out # output file name
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --error=job_outputs/%s%%j.err  # error file name

set -x
cd ${SLURM_SUBMIT_DIR}

module purge
module load pytorch-gpu/py3/1.8.1
module load cmake
module load cuda

python ./one_expe.py %s
''' % (name, name, name, args_string))


expe_name = 'compare_convergence'
n_tries = 1
#
iterations = [3]
print('Launching on %d nodes' % (n_tries * len(iterations)))

for i in range(n_tries):
    for n_it in iterations:
        args_str = '--dataset imdb --vocab_file wiki.vocab --tokenizer sentencepiece' \
                   ' --pretrained_model wiki.model --n_it %d --seed %s' % (n_it, i)
        name = 'sink_' + str(n_it)
        write_script(expe_name, name, args_str)
        os.system('sbatch %s.sh' % expe_name)
        time.sleep(.1)

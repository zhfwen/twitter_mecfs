#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --account=**user**
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --mail-user=**email**
#SBATCH --mail-type=ALL
#SBATCH --output=./slurm/%A_%a.out
#SBATCH --array=0-183  # 92 chunks per dataset * 2 logits

module load StdEnv/2023 python/3.11.5 arrow/15.0.1
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Determine the chunk number and logit value
chunk_id=$((SLURM_ARRAY_TASK_ID / 2))  # Each chunk ID repeats twice (for logits 0 and 1)
logit=$((SLURM_ARRAY_TASK_ID % 2))     # 0 or 1 for each chunk

chunk_number=$((chunk_id % 92 ))   # Which chunk (1 to 92)


# Run the Python script with the selected dataset and logit
python shapley.py $chunk_number $logit
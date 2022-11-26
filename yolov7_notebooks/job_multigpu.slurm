#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100m40:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name="viproject_multigpu_njc"
#SBATCH --output=yolov7_multigpu_njc.out
#SBATCH --mail-user=eivind.kohmann@gmail.com
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"


cd ${SLURM_SUBMIT_DIR}/yolov7

module purge
module load Python/3.8.6-GCCcore-10.2.0

pip install -r requirements.txt --user -q
pip install wandb --user -q

wandb login "c20e2fa019f3958082d55153ba1463c147434dec"

python3 -m torch.distributed.launch --nproc_per_node=2 train.py --batch 32 --name yolov7_multigpu_njc  --epochs 50 --data ../roadcrack.yaml --weights 'yolov7_training.pt' --device 0,1 --workers 24 --img 1000 --cfg cfg/training/yolov7.yaml --hyp data/hyp.scratch.custom.yaml

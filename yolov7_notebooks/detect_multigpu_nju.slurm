#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100m40:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name="viproject_multigpu_prediction_nju"
#SBATCH --output=yolov7_big_prediction_nju.out
#SBATCH --mail-user=eivinkoh@stud.ntnu.no
#SBATCH --mail-type=ALL


cd ${SLURM_SUBMIT_DIR}/yolov7

module purge
module load Python/3.8.6-GCCcore-10.2.0

pip install -r requirements.txt --user -q


python3 detect.py --weights runs/train/yolov7_multigpu_njc/weights/best.pt --conf 0.1 --img-size 1024 --source ../../../../../../projects/vc/courses/TDT17/2022/open/RDD2022/Norway/test/images/ --name predictions_multigpu_nju --save-txt --save-conf --device 0,1
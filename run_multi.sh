#!/bin/sh

# run parameters
run_name='train_multi'
logfile=./logs/train_multi.log
train_script='train_multi.py'
ngpus=2
port=9995
epochs=150
epochs_per_test=1
batch_size=128
num_workers=4

# check if the log file already exists and remove it
if [[ -f $logfile ]]; then
    echo $filename
    /bin/rm -v $logfile
fi

# command to run the training script
cmd="python -m torch.distributed.launch --nproc_per_node=$ngpus --master_port=$port $train_script --epochs=$epochs --run_name=$run_name --num_workers=$num_workers --batch_size=$batch_size | tee -a $logfile"

# run the training script
echo $cmd > $logfile
eval $cmd

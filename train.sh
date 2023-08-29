#!/bin/sh
PARTITION=Segmentation

GPU_ID=0
dataset=pascal # pascal coco
exp_name=split0
arch=FCA_HLP
net=vgg

now=$(date +"%Y-%m-%d_%X")
exp_dir=exp/${dataset}/${arch}/${exp_name}/${net}/${now}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
show_dir=${exp_dir}/show
config=config/${dataset}/${dataset}_${exp_name}_${net}.yaml
model_dir=model/${arch}.py
mkdir -p ${snapshot_dir} ${result_dir} ${show_dir}
cp train.sh train.py ${config} ${model_dir}  ${exp_dir}

echo ${arch}
echo ${config}



CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train.py \
        --config=${config} \
        --arch=${arch} \
        --exp_dir=${exp_dir} \
        --snapshot_dir=${snapshot_dir} \
        --result_dir=${result_dir} \
        --show_dir=${show_dir} \
        2>&1 | tee ${result_dir}/train-$now.log



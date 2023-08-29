#!/bin/sh
PARTITION=Segmentation

GPU_ID=0
dataset=pascal # pascal coco fss
exp_name=split1

arch=FCA_HLP
net=vgg # vgg resnet50 resnet101


now=$(date +"%Y-%m-%d_%X")
exp_dir=exp/${dataset}/${arch}/${exp_name}/${net}/${now}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result/test/${now}
show_dir=${exp_dir}/show
config=config/${dataset}/${dataset}_${exp_name}_${net}.yaml
mkdir -p ${snapshot_dir} ${result_dir} ${show_dir}
cp test.sh test.py ${config} ${result_dir}

echo ${arch}
echo ${config}

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u test.py \
        --config=${config} \
        --arch=${arch} \
        --exp_dir=${exp_dir} \
        2>&1 | tee ${result_dir}/test-$now.log
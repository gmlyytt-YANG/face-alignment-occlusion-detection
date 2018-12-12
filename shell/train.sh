#!/bin/bash

# param init
show=""
content=""
epochs=""
bs=""
lr=""
mode=""
phase=""
feature=""
label_ext=""
model_type=""
loss_name=""

# param analyse
while [ -n "$1" ]
do
    case "$1" in
    "-s")
        show="$2"
        ;;
    "-c")
        content="$2"
        ;;
    "-e")
        epochs="$2"
        ;;
    "-bs")
        bs="$2"
        ;;
    "-lr")
        lr="$2"
        ;;
    "-m")
        mode="$2"
        ;;
    "-p")
        phase="$2"
        ;;
    "-f")
        feature="$2"
        ;;
    "-le")
        label_ext="$2"
        ;;
    "-mt")
        model_type="$2"
        ;;
    "-ln")
        loss_name="$2"
        ;;
    esac
    shift
done

# run cmd
if [ ${show} = "nohup" ];then
    nohup python train.py -s ${show} -c ${content} -e ${epochs}\
        -bs ${bs} -lr ${lr} -m ${mode} -p ${phase} -f ${feature}\
        -le ${label_ext} -mt ${model_type} -ln ${loss_name}\
    > logs/epochs=${epochs}_bs=${bs}_initlr=${lr}_mode=${mode}_phase=${phase}_content=${content}.log 2>&1 &
else
    python train.py -s ${show} -c ${content} -e ${epochs}\
        -bs ${bs} -lr ${lr} -m ${mode} -p ${phase} -f ${feature}\
        -le ${label_ext} -mt ${model_type} -ln ${loss_name}
fi

#!/bin/bash

# param init
epochs=""
bs=""
lr=""
mode=""
show=""
phase=""
content=""

# adaptor param init
epochs1=""
bs1=""
lr1=""
epochs2=""
bs2=""
lr2=""

# param analyse
while [ -n "$1" ]
do
    case "$1" in
    "-s")
        show="$2"
        ;;
    "-p")
        phase="$2"
        if [ ${phase} = "pre" ];then
            break
        fi
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
    "-e1")
        epochs1="$2"
        ;;
    "-bs1")
        bs1="$2"
        ;;
    "-lr1")
        lr1="$2"
        ;;
    "-e2")
        epochs2="$2"
        ;;
    "-bs2")
        bs2="$2"
        ;;
    "-lr2")
        lr2="$2"
        ;;
    "-m")
        mode="$2"
        ;;
    esac
    shift
done

# run cmd
if [ ${show} = "nohup" ];then
    if [ ${phase} = "pre" ];then
        nohup python preprocess.py > \
        logs/preprocess.log 2>&1 &
    elif [ ${phase} = "adaptor" ];then
        nohup python adaptor.py -e1 ${epochs1} -bs1 ${bs1} -lr1 ${lr1} -e2 ${epochs2} -bs2 ${bs2} -lr2 ${lr2}\
        > logs/adaptor.log 2>&1 &
    else
        nohup python train.py -e ${epochs} -bs ${bs} -lr ${lr} -m ${mode} -p ${phase} -c ${content} -dAgg\
        > logs/epochs=${epochs}_bs=${bs}_initlr=${lr}_mode=${mode}_phase=${phase}_content=${content}.log 2>&1 &
    fi
else
    if [ ${phase} = "pre" ];then
        python preprocess.py
    elif [ ${phase} = "adaptor" ];then
        python adaptor.py -e1 ${epochs1} -bs1 ${bs1} -lr1 ${lr1} -e2 ${epochs2} -bs2 ${bs2} -lr2 ${lr2}
    else
        python train.py -e ${epochs} -bs ${bs} -lr ${lr} -m ${mode} -p ${phase} -c ${content} -dAgg
    fi
fi

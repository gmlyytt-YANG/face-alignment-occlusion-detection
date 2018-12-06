#!/bin/bash

# param init
epochs=""
bs=""
lr=""
mode=""
show=""
phase=""

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
        if [ ${phase} = "pre" ] || [ ${phase} = "adaptor" ];then
            break
        fi
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
    "-h1")
        epochs1="$2"
        ;;
    "-p1")
        bs1="$2"
        ;;
#    "-LR1")
#        lr1="$2"
#        ;;
#    "-E2")
#        epochs2="$2"
#        ;;
#    "-BS2")
#        bs2="$2"
#        ;;
#    "-LR2")
#        lr2="$2"
#        ;;
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
        # nohup python adaptor.py -E1 ${epochs1} -BS1 ${bs1} -LR1 ${lr1} -E2 ${epochs2} -BS2 ${bs2} -LR2 ${lr2}\
        nohup python adaptor.py -h1 ${epochs1} -p1 ${bs1}\
        > logs/adaptor.log 2>&1 &
    else
        nohup python train.py -e ${epochs} -bs ${bs} -lr ${lr} -m ${mode} -p ${phase} -dAgg\
        > logs/epochs=${epochs}_bs=${bs}_initlr=${lr}_mode=${mode}_phase=${phase}.log 2>&1 &
    fi
else
    if [ ${phase} = "pre" ];then
        python preprocess.py
    elif [ ${phase} = "adaptor" ];then
        python adaptor.py -h1 ${epochs1} -p1 ${bs1}
    else
        python train.py -e ${epochs} -bs ${bs} -lr ${lr} -m ${mode} -p ${phase} -dAgg
    fi
fi

#!/bin/bash

# param init
epochs=""
bs=""
lr=""
mode=""
show=""
phase=""

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
        nohup python adaptor.py > logs/adaptor.log 2>&1 & 
    else 
        nohup python train.py -e ${epochs} -bs ${bs} -lr ${lr} -m ${mode} -p ${phase} -dAgg\
        > logs/epochs=${epochs}_bs=${bs}_initlr=${lr}_mode=${mode}_phase=${phase}.log 2>&1 &
    fi
else
    if [ ${phase} = "pre" ];then
	    python preprocess.py
    elif [ ${phase} = "adaptor" ];then
        python adaptor.py
    else
        python train.py -e ${epochs} -bs ${bs} -lr ${lr} -m ${mode} -p ${phase} -dAgg
    fi
fi

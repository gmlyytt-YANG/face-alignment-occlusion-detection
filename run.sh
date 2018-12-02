#!/bin/bash

# param init
epochs=""
bs=""
lr=""
mode=""
show=""
phase=""

# param analyse
while [[ -n "$1" ]]
do
    case "$1" in
    "-p")
	phase="$2"
	if [[ ${phase} = "pre" ]];then
	    break
	fi
	;;
    "-s")
	show="$2"
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

if [[ ${show} = "nohup" ]];then
    if [[ ${phase} = "pre" ]];then
        nohup python preprocess.py > \
        logs/epochs=${epochs}_bs=${bs}_initlr=${lr}_mode=${mode}_pre.log 2>&1 &
    else if [[ ${phase} = "occlu" ]];then
        nohup python bootstrap.py -e ${epochs} -bs ${bs} -lr ${lr} -m ${mode} \
        > logs/epochs=${epochs}_bs=${bs}_initlr=${lr}_mode=${mode}_occlu.log 2>&1 &
    else

else
    python occlu_bootstrap.py -e ${epochs} -bs ${bs} -lr ${lr} -m ${mode}
fi

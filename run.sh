#!/bin/bash

# param init
epochs=""
bs=""
lr=""
mode=""
show=""

# param analyse
while [ -n "$1" ]           
do
    case "$1" in
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
if [ ${show} = "nohup" ];then
    nohup python occlu_bootstrap.py -e ${epochs} -bs ${bs} -lr ${lr} -m ${mode} \
	> logs/epochs=${epochs}_bs=${bs}_initlr=${lr}_mode=${mode}.log 2>&1 &
else
    python occlu_bootstrap.py -e ${epochs} -bs ${bs} -lr ${lr} -m ${mode}
fi

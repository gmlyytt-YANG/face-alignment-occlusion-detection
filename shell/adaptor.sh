#!/bin/bash

# adaptor param init
show=""
label_ext=""
feature=""
epochs1=""
bs1=""
lr1=""
mt1=""
ln1=""
c1=""
epochs2=""
bs2=""
lr2=""
mt2=""
ln2=""
c2=""

# param analyse
while [ -n "$1" ]
do
    case "$1" in
    "-s")
        show="$2"
        ;;
    "-le")
        label_ext="$2"
        ;;
    "-f")
        feature="$2"
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
    "-mt1")
        mt1="$2"
        ;;
    "-ln1")
        ln1="$2"
        ;;
    "-c1")
        c1="$2"
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
    "-mt2")
        epochs2="$2"
        ;;
    "-ln2")
        ln2="$2"
        ;;
    "-c2")
        c2="$2"
        ;;
    esac
    shift
done


# run cmd
if [ ${show} = "nohup" ];then
    nohup python adaptor.py -le ${label_ext} -f ${feature} -e1 ${epochs1}\
         -bs1 ${bs1} -lr1 ${lr1} -mt1 ${mt1} -ln1 ${ln1} -c1 ${c1} -e1 ${epochs1} -bs1 ${bs1} -lr1 ${lr1}\
         -mt1 ${mt1} -ln1 ${ln1} -c1 ${c1}\
    > logs/epochs1=${epochs1}_bs1=${bs1}_lr1=${lr1}_content1=${c1}_epochs2=${epochs2}_bs2=${bs2}_lr2=${lr2}_content2=${c2}.log 2>&1 &
else
    python adaptor.py -le ${label_ext} -f ${feature} -e1 ${epochs1}\
         -bs1 ${bs1} -lr1 ${lr1} -mt1 ${mt1} -ln1 ${ln1} -c1 ${c1} -e1 ${epochs1} -bs1 ${bs1} -lr1 ${lr1}\
         -mt1 ${mt1} -ln1 ${ln1} -c1 ${c1}
fi

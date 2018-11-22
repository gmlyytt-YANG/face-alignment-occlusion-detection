if [ "$1" = "nohup" ];then
    nohup python bootstrap.py -e "$2" -bs "$3" -lr "$4" > logs/epochs=$2_bs=$3_initlr=$4.log 2>&1 &
else
    python bootstrap.py -e "$2" -bs "$3" -lr "$4"
fi

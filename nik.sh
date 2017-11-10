#!/bin/bash

USER="maf388"
RMPT=52698
PORT=8888

if [ ! -z $1 ]; then
    if [ ! -z $2 ]; then
        ssh -t -R $RMPT:localhost:$RMPT -L $PORT:localhost:$PORT $USER@nikola-compute0$1.coecis.cornell.edu "export CUDA_VISIBLE_DEVICES=$2; bash"
    else
        ssh -R $RMPT:localhost:$RMPT -L $PORT:localhost:$PORT $USER@nikola-compute0$1.coecis.cornell.edu
    fi
    exit 0
fi

CMD="nvidia-smi --format=csv --query-gpu=memory.used>tmp.txt;\
sed '1d' tmp.txt>>gpumem.txt;\
rm tmp.txt; exit"

for n in {1..6}
do
    #echo $n
    ssh $USER@nikola-compute0$n.coecis.cornell.edu $CMD
done
rsync --remove-source-files $USER@nikola-compute01.coecis.cornell.edu:\
/home/$USER/gpumem.txt ~/gpumem.txt

N=1
M=0
NOPEN=-1
MOPEN=-1
while read -r line
do
    memUsed=($line)
    echo "Machine-$N GPU-$M usage: $memUsed MB"

    if [ $memUsed -le 100 ]; then
        NOPEN=$N
        MOPEN=$M
    fi

    M=$((M+1))
    if [ $M -ge 4 ]; then
        N=$((N+1))
        M=0
    fi
done <<< "$(cat ~/gpumem.txt)"
#rm ~/gpumem.txt
echo ""
if [ $NOPEN -ge 0 ]; then
    echo "NIKOLA-$NOPEN GPU $MOPEN is available"
else
    echo "All machines are busy."
fi

exit 0

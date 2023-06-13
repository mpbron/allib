#!/bin/bash
for dataset in $(cat $2); do 
    f="$1/$dataset"
    for m in $(cat $4); do
        for ((i = 1; i <= $5; i++)); do
            echo "python -m allib -m Review -d $f -t $3 -e $m -r $i"
        done
    done
done

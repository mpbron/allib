#!/bin/bash
# Our custom function
echo "Reading path $1 for Review datasets"
echo "Selecting datasets specified $2"
echo "Writing results in $3"
echo "Running experiments specified in $4"
echo "Repeating each experiment $5 times"
echo "Running on $6 parallel cores"
./create_jobs.sh $1 $2 $3 $4 $5| parallel -j $6
#!/bin/bash
# Our custom function
find $1 -type f | parallel -j $3 ./benchmark.sh $2
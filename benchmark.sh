#!/bin/bash
# Our custom function
run_benchmark(){
  python -m allib ./datasets/Nagtegaal_2019.csv ./results NaiveBayesEstimator TfIDF5000
}
for i in {1..10}
do
	run_benchmark
done
 
## Put all cust_func in the background and bash 
## would wait until those are completed 
## before displaying all done message
wait 
echo "All done"
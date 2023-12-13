#!/bin/bash

dims=(100 500 1000)

output_file=results.txt
rm $output_file
g++-13 -fopenmp parallel.cpp -o par.out
g++-13 -fopenmp ../sequential.cpp -o seq.out

for dim in "${dims[@]}"; do
    ./seq.out $dim | tee -a $output_file
    ./par.out $dim | tee -a $output_file
done

rm par.out
rm seq.out
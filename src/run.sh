#!/bin/bash

dims=(160 640 1280)

output_file=seq_res.txt

g++-13 -fopenmp sequential.cpp -o seq.out

for dim in "${dims[@]}"; do
    ./seq.out $dim | tee -a $output_file
done

rm seq.out
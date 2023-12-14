#!/bin/bash

procs=(2 4 8 16)
dims=(160 640 1280)

output_file=results.txt
prog_name=parallel

rm "$output_file"

mpic++ -w "$prog_name".cpp -o "$prog_name".o

if [ $? -ne 0 ]; then
    exit
fi

for dim in "${dims[@]}"; do
    for proc in "${procs[@]}"; do
        echo "-- Start experiment with N=$dim and $proc processes"
        mpirun -np "$proc" "$prog_name".o "$dim" "$output_file"
        echo -e "-- Completed!\n"
    done
done

rm "$prog_name".o
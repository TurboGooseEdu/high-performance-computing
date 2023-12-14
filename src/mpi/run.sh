#!/bin/bash

prog_name=parallel

mpic++ "$prog_name".cpp -o "$prog_name".o -w
mpirun -np 2 "$prog_name".o 
rm "$prog_name".o
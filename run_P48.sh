#!/bin/bash
#SBATCH --job-name=Assign2_P48
#SBATCH --output=output_P48.txt
#SBATCH --error=error_P48.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=00:10:00

module load compiler/oneapi-2024/mpi

echo "Running P=48 Experiment"

d=7
ppn=48
px=6
py=4
pz=2
T=5
seed=1000
F=2
iso=500.0

for size in 120 240; do
    echo "--- Configuration: size=$size ---"
    nx=$size
    ny=$size
    nz=$size
    for run in {1..5}; do
        echo "---Run${run}---"
        mpirun -np 48 ./src $d $ppn $px $py $pz $nx $ny $nz $T $seed $F $iso
    done
done
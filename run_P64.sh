#!/bin/bash
#SBATCH --job-name=Assign2_P64
#SBATCH --output=output_P64.txt
#SBATCH --error=error_P64.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:10:00

module load compiler/oneapi-2024/mpi

echo "Running P=64 Experiment"

d=7
ppn=32
px=4
py=4
pz=4
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
        mpirun -np 64 ./src $d $ppn $px $py $pz $nx $ny $nz $T $seed $F $iso
    done
done
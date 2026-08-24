#!/bin/bash
#SBATCH --job-name=Assign2_P32
#SBATCH --output=output_P32.txt
#SBATCH --error=error_P32.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:10:00

module load compiler/oneapi-2024/mpi

echo "Running P=32 Experiment"

d=7
ppn=32
px=4
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
        mpirun -np 32 ./src $d $ppn $px $py $pz $nx $ny $nz $T $seed $F $iso
    done
done
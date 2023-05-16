#!/bin/sh
#SBATCH --job-name=L666_NS # Job name
#SBATCH --ntasks-per-node=12  # Number of tasks per node
#SBATCH --nodes=1
#SBATCH --time=48:00:00 # Time limit hrs:min:sec
#SBATCH --partition=debug

echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
#echo "CPU bind Info  = $SLURM_CPU_BIND"    
echo "CPU on node  = $SLURM_CPUS_ON_NODE"         

cd $SLURM_SUBMIT_DIR
module load openmpigcc9.2
module load gcc9.2
# module load cuda11gcc9.2
module load mkl/2021.2.0
#module load oneapi
#module load intelmkl20
#srun -n 2 --mpi=pmi2   /shared/LAMMPS/lammps/src/lmp_kokkos_mpi_only -k on -sf kk -in melt_quench_update > Si24C5H36_2fold_annealing.out
export LD_LIBRARY_PATH="$HOME/.local/lib64:/apps/anaconda/lib:$LD_LIBRARY_PATH"
mpirun -n $SLURM_NTASKS python3 Normal_Stress_Parallel.py > /dev/null
# mpirun -n $SLURM_NTASKS python3 Normal_Stress_Parallel.py > /dev/null


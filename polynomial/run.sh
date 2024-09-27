export OMP_NUN_THREADS=576
export OMP_PROC_BIND=True
export OMP_PLACES=cores 
make clean
make
./polynomial test2.conf

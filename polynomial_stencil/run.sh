export OMP_NUN_THREADS=576
export OMP_PROC_BIND=True
export OMP_PLACES=cores 
make clean
make
clang runner.c -o runner
./runner
#./polynomial_stencil test2.conf

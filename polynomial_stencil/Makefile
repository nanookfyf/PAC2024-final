#all:

all: polynomial_stencil
#	clang -march=armv8-a+sve -std=c11 -fopenmp -O3 -g main.c polynomial_stencil.c -o polynomial 
main.o: main.c
	clang -c -march=armv9-a+sve2 -std=c11 -fopenmp -O3 -g main.c -L/home/PAC20241694/package/numa/usr/lib64 -I/home/PAC20241694/package/numa/usr/include -flto

polynomial_stencil.o: polynomial_stencil.c
	clang -c -march=armv9-a+sve2 -std=c11 -fopenmp -O3 -g polynomial_stencil.c -I/home/PAC20241694/package/numa/usr/include -flto

polynomial_stencil: polynomial_stencil.o main.o
	@clang++ polynomial_stencil.o main.o -L/home/PAC20241694/package/numa/usr/lib64 -lnuma -o polynomial_stencil -fopenmp -flto -O3
#-L. -lmalloc -Wl,--wrap=malloc

clean: 
	rm *.o polynomial_stencil


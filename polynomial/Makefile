all: polynomial
main.o: main.c
	clang -c -march=armv9-a+sve2 -std=c11 -fopenmp -O3 -fno-fast-math -g main.c -I/home/PAC20241694/package/numa/usr/include -flto

Pfinal.o: Pfinal.c
	clang -c -march=armv9-a+sve2 -std=c11 -fopenmp -O3 -fno-fast-math -g Pfinal.c -I/home/PAC20241694/package/numa/usr/include -flto

polynomial: Pfinal.o main.o
	@clang++  Pfinal.o main.o  -L/home/PAC20241694/package/numa/usr/lib64 -lnuma -o polynomial -fopenmp -flto -O3 
clean: 
	rm *.o polynomial


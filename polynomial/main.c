#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>

#include "polynomial.h"

const long MAX_NX = (8L * 1000L * 1000L * 1000L);
const double MAX_DIFF = 1e-8;
const int ITER_TIMES = 5;

/*void polynomial_verify(double *fa, double *f, long nx, double p[], int term)
{
    double x;

    long i;
    int j;
    for (i = 0; i < nx; i++) {
        x = f[i];
        fa[i] = p[term - 1];
        for (j = term - 2; j >= 0; j--) {
            fa[i] *= x;
            fa[i] += p[j];
        }
    }
}*/

void polynomial_verify(double *fa, double *f, long nx, double p[], int term)
{
    double x;
    long i;
    int thread = 0;
    int j;
    int parallel_thread_num = 576;

    
    //int reg_size = svcntd(); // 向量寄存器大小
    
    //int reg_num = nx / reg_size; // 向量寄存器数量
    //int tail = nx % reg_size;

    

    //int ave_times = reg_num / parallel_thread_num; // 线程中 迭代几次 向量寄存器
    int thread_block = nx/parallel_thread_num;  // 每个线程待处理的点



    #pragma omp parallel for num_threads(parallel_thread_num) schedule(static) private(i,j,x) 
    for ( thread = 0; thread < parallel_thread_num; thread++) {
       
        for (int k=0;k<thread_block;k++){
                i = thread * thread_block + k;
                x = f[i];
                fa[i] = p[term - 1];
                for (j = term - 2; j >= 0; j--) {
                    fa[i] *= x;
                    fa[i] += p[j];
                }
            
        }  
    }
    
    for(i = thread_block*parallel_thread_num;i<nx;i++) {
        x = f[i];
        fa[i] = p[term - 1];
        for (j = term - 2; j >= 0; j--) {
            fa[i] *= x;
            fa[i] += p[j];
        }
    }


}



void print_array(double *fa, long nx)
{
    long i;
    printf("start print array\n");
    for (i = 0; i < nx; i++) {
        printf("%f ", fa[i]);
    }
    printf("\n");
}

int CompareResults(double *se, double *seOpt, long size)
{
    long i;
    for (i = 0; i < size; i++) {
        double diff = se[i] - seOpt[i];
        if ((diff > MAX_DIFF) || (diff < -MAX_DIFF)) {
            printf("compute error at %ld, result: %.20lf %.20lf, diff %.20lf\n", i, se[i], seOpt[i], diff);
            return -1;
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    long nx;
    int i, j;
    double max_num = 2.0;
    struct timeval start, stop;

    nx = MAX_NX;

    double *f = malloc(nx * sizeof(double));
    double *fa = malloc(nx * sizeof(double));
    double *fb = malloc(nx * sizeof(double));

    FILE *fp = fopen(argv[1], "r");
    double compute_time = 0.0;
    double iter_time = 0.0;

    int test = 1;
    int term;
    double p[15];
    int missed = 0;
    int iter;
    while (fscanf(fp, "%ld", &nx) != EOF) {

        if(nx > MAX_NX || nx <=0 ) {
            printf("nx value error.\n");
            return -2;
        }

        srand(test);

        fscanf(fp, "%d", &term);

        for (j = 0; j < term; j++) {
            fscanf(fp, "%lf", &p[j]);
        }

        printf("test %d , nx %ld, term %d, p0: %lf, p[-1]: %lf\n", test, nx, term, p[0], p[term - 1]);

        for (iter = 1; iter <= ITER_TIMES; iter++) {
            srand(iter);
            for (j = 0; j < nx; j++) {
                f[j] = (double)(rand() / (double)(RAND_MAX / max_num));
            }
            gettimeofday(&start, (struct timezone *)0);
            polynomial(fa, f, nx, p, term);

            gettimeofday(&stop, (struct timezone *)0);
            polynomial_verify(fb, f, nx, p, term);

            iter_time = (double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) * 1.e-6;
            compute_time += iter_time;
            printf("iteration %d compute time:%10.6f sec\n", iter, iter_time);

            missed += CompareResults(fa, fb, nx);
        }

        if (missed) {
            if (missed == -ITER_TIMES) {
                printf("compute error!\n");
            }
            printf("part of compute result varifed fail, optimizing instable!\n");
            return -1;
        } else {
            printf("compute results are verifed success!\n");
        }

        test++;
    }
    printf("all test cases compute time is %lf sec.\n", compute_time);

    return 0;
}

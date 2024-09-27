#define _GNU_SOURCE

#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <arm_sve.h>
#include <unistd.h>
#include <string.h>
#include "numaif.h"

/*void *__wrap_malloc(size_t size) {
    void* ptr = aligned_alloc(4096,size) ;
    //posix_memalign((void**)&ptr,4096, size);
    //return ptr;
    const static long MAX_NX = (8L * 1000L * 1000L * 1000L);
    if (size == MAX_NX * sizeof(double)) {
	    //write(2, "MAGIC!\n", 7);
	long long mem_per_numa = 528000000 * sizeof(double) / 4;
	mem_per_numa &= (~0xfff);
	static unsigned long mask1 = 1, mask2 = 2, mask3 = 4, mask4 = 8;
        mbind(ptr, mem_per_numa, MPOL_BIND, &mask1, 4, MPOL_MF_MOVE);
        mbind(ptr + mem_per_numa, mem_per_numa, MPOL_BIND, &mask2, 4, MPOL_MF_MOVE);
        mbind(ptr + mem_per_numa * 2, mem_per_numa, MPOL_BIND, &mask3, 4, MPOL_MF_MOVE);
        mbind(ptr + mem_per_numa * 3, mem_per_numa, MPOL_BIND, &mask4, 5, MPOL_MF_MOVE);

    }
    static int omp_init = 1;
    if(omp_init) {
#pragma omp parallel for num_threads(576) schedule(static)
            for(int i = 0; i < 576; i++) {
                    getpid();
            }
            omp_init = 0;
    }
    return ptr;
}*/
static void bind_region(void* ptr, unsigned long mem_per_numa) {
	ptr = (void*) (((unsigned long long) ptr) & (~0xfffull));
	mem_per_numa &= (~0xfff);
	static unsigned long mask1 = 1, mask2 = 2, mask3 = 4, mask4 = 8;
        mbind(ptr, mem_per_numa, MPOL_BIND, &mask1, 5, MPOL_MF_MOVE);
        mbind(ptr + mem_per_numa, mem_per_numa, MPOL_BIND, &mask2, 5, MPOL_MF_MOVE);
        mbind(ptr + mem_per_numa * 2, mem_per_numa, MPOL_BIND, &mask3, 5, MPOL_MF_MOVE);
        mbind(ptr + mem_per_numa * 3, mem_per_numa, MPOL_BIND, &mask4, 5, MPOL_MF_MOVE);
}


void polynomial(double *fa, double *f, long nx, double p[], int term)
{
    double x;
    long i;
    int thread = 0;
    int j;

    static int init = 0;
	if (init == 0) {
	    //	scanf("%d", &init);
		//printtime();
		bind_region(fa, nx * sizeof(double) / 4);
		bind_region(f, nx * sizeof(double) / 4);
		init = 1;
		//printtime();
	}
    
    int parallel_thread_num = 0;

    if (nx>=576){
        parallel_thread_num = 576;
    }else{
        parallel_thread_num = nx;
    }
    int reg_size = svcntd(); // 向量寄存器大小
    int thread_block = nx / parallel_thread_num; //每个线程待处理的点
    //int tail = nx % parallel_thread_num;  
    int ave_times = thread_block /reg_size; // 线程中 迭代几次 向量寄存器
    //int block_tail = thread_block % 8;

    #pragma omp parallel for num_threads(parallel_thread_num) schedule(static) private(i,j,x) 
    for ( thread = 0; thread < parallel_thread_num; thread++) {
        // 向量寄存器
        //x = f[i];
        //fa[i] = p[term - 1];
        //for (j = term - 2; j >= 0; j--) {
          //  fa[i] *= x;
            //fa[i] += p[j];
        //}
        for (int k=0;k<ave_times;++k){
            i = thread * thread_block + k*reg_size;
            svfloat64_t vx = svld1(svptrue_b64(), &f[i]);
            svfloat64_t vf = svdup_f64(p[term - 1]);

            for (j = term - 2; j >= 0; --j) {
                //vf *= x;
                //fa[i] += p[j];
                vf = svmla_x(svptrue_b64(), svdup_f64(p[j]), vf, vx);
            }

            svst1(svptrue_b64(), &fa[i], vf);
        }
        for(int k=ave_times*reg_size;k<thread_block;++k){
            i = thread * thread_block + k;
            x = f[i];
            fa[i] = p[term - 1];
            for (j = term - 2; j >= 0; --j){
                fa[i] *= x;
                fa[i] += p[j];
            }
        }
    }

    for(i = thread_block*parallel_thread_num;i<nx;++i) {
        x = f[i];
        fa[i] = p[term - 1];
        for (j = term - 2; j >= 0; --j) {
            fa[i] *= x;
            fa[i] += p[j];
        }
    }


}


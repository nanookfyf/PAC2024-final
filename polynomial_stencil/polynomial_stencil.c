#define _GNU_SOURCE
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <arm_sve.h>
#include <unistd.h>
#include <string.h>
#include <numaif.h>
int first=0;



static void bind_region(void* ptr, unsigned long mem_per_numa) {
	ptr = (void*) (((unsigned long long) ptr) & (~0xfffull));
	mem_per_numa &= (~0xfff);
	static unsigned long mask1 = 1, mask2 = 2, mask3 = 4, mask4 = 8;
        mbind(ptr, mem_per_numa, MPOL_BIND, &mask1, 5, MPOL_MF_MOVE);
        mbind(ptr + mem_per_numa, mem_per_numa, MPOL_BIND, &mask2, 5, MPOL_MF_MOVE);
        mbind(ptr + mem_per_numa * 2, mem_per_numa, MPOL_BIND, &mask3, 5, MPOL_MF_MOVE);
        mbind(ptr + mem_per_numa * 3, mem_per_numa, MPOL_BIND, &mask4, 5, MPOL_MF_MOVE);
}

void polynomial_stencil(double *fa, double *f, long nx, double p[], int term)
{
	static volatile do_exit = 1;
//	if (do_exit)exit(0);
	static int init = 0;
	if (init == 0) {
	//	scanf("%d", &init);
		//printtime();
		bind_region(fa, nx * sizeof(double) / 4);
		bind_region(f, nx * sizeof(double) / 4);
		init = 1;
		//printtime();
	}
    int idx = term / 2;
    long i;
    int j;
    for(i=-1;i>=-idx;--i){
        fa[i]=0;
        fa[nx-i-1]=0;
    }
    double x;

    int parallel_thread_num = 576;
       
    long thread_box = nx/parallel_thread_num;
   
    long u=0;
    int thread = 0;
    #pragma omp parallel for num_threads(parallel_thread_num) schedule(static) private(j,u)
    for (thread = 0; thread < parallel_thread_num; ++thread) {
	   int nnode, ncpu;
	  getcpu(&ncpu, &nnode);
	  //printf("%d, %d, %d\n", ncpu, nnode, thread);
        //if(i==omp_get)
        //double *table = malloc((idx-1)*term*sizeof(double));
        svfloat64_t vec_array;
        svfloat64_t next1,next2,next3,next4; //流水
        svbool_t pg = svptrue_b64(); //8个double 并行
        //svbool_t pg1 = svwhilelt_b64((int64_t)0, (int64_t)term%8);
        svfloat64_t vec_array1,vec_pow; //
        svfloat64_t result=svdup_n_f64(0.0);

        long ii = thread*thread_box;
        u=0;
        int powcnt = idx;
        if(thread_box>=8){
            /*第一次并行处理（初始化向量寄存器的）*/
            vec_array1 = svld1(pg, &f[ii-idx]);
            result = svdup_n_f64(1.0);

            vec_pow = vec_array1; 
            //
            while (powcnt>0)
            {
                /* code */
                if(powcnt&1) result = svmul_f64_z(pg, vec_pow, result);
                vec_pow=  svmul_f64_z(pg, vec_pow, vec_pow);
                powcnt >>= 1;
            }
            //
            //for(int k = 1; k < idx; ++k)
              //  result = svmul_f64_z(pg, vec_array1, result);
            result = svmul_f64_z(pg, svdup_f64(p[0]), result);
            next1=svld1(pg,&f[ii-idx+8]);//预取
            next2=svld1(pg,&f[ii-idx+16]);//预取
            next3=next1;//保留便于再次获取
            next4=next2;
            for (int m=1;m<term;++m){
                //左移
                vec_array1 = svext_f64(vec_array1,next1, (int64_t)1);
                next1 = svext_f64(next1,next2, (int64_t)1);
                next2 = svext_f64(next2,next2, (int64_t)1);
                //vec_array = svdup_n_f64(1.0);
                vec_array = vec_array1;
               

                for(int k = 1; k < abs(m - idx); ++k)
                    vec_array = svmul_f64_z(pg, vec_array1, vec_array);
                //类似滚动数组的模式记录结果
                result = svmla_x(pg, result, vec_array, svdup_f64(p[m]));
                
            }
            svst1(pg, &(fa[ii]), result);
        /*并行处理*/
        for(u=8;u<thread_box-8;u+=8){
            ii = thread*thread_box +u ;
            vec_array1 = next3;
            //vec_array1 =result;
            /*********头部高次项用快速幂***********/
            result = svdup_n_f64(1.0);
            vec_pow = vec_array1; 
            powcnt = idx;
            while (powcnt>0)
            {
                /* code */
                if(powcnt&1) result = svmul_f64_z(pg, vec_pow, result);
                vec_pow=  svmul_f64_z(pg, vec_pow, vec_pow);
                powcnt >>= 1;
            }
            //for(int k = 1; k < idx; ++k)
                //result = svmul_f64_z(pg, vec_array1, result);
            result = svmul_f64_z(pg, svdup_f64(p[0]), result);

            next1=next4;
            next2=svld1(pg,&f[ii-idx+16]);
            next3=next1;
            next4=next2;

            for (int m=1;m<term-1;++m){
                vec_array = svext_f64(vec_array1,next1, (int64_t)1);
                next1 = svext_f64(next1,next2, (int64_t)1);
                next2 = svext_f64(next2,next2, (int64_t)1);
                vec_array1 =vec_array;
                for(int k = 1; k < abs(m - idx); ++k)
                    vec_array = svmul_f64_z(pg, vec_array1, vec_array);
                result = svmla_x(pg, result, vec_array, svdup_f64(p[m]));
                
            }
            /*********尾部高次项用快速幂***********/
            vec_array1 = svext_f64(vec_array1,next1, (int64_t)1);
            vec_array = svdup_n_f64(1.0) ;
            vec_pow = vec_array1;
            powcnt = abs(term-1 - idx);
            while (powcnt>0)
                {
                    
                    if(powcnt&1) vec_array = svmul_f64_z(pg, vec_pow, vec_array);
                    vec_pow=  svmul_f64_z(pg, vec_pow, vec_pow);
                    powcnt >>= 1;
                }
                //for(int k = 1; k < abs(term-1 - idx); ++k)
                  //  vec_array = svmul_f64_z(pg, vec_array1, vec_array);
            result = svmla_x(pg, result, vec_array, svdup_f64(p[term-1]));
            
            svst1(pg, &(fa[ii]), result);
        } 
        
        }



        //tail
        ii = thread*thread_box +u ;
        svbool_t newpg = svwhilelt_b64((int64_t)0, (int64_t)thread_box%8);
        result = svdup_n_f64(0.0);
        for (int m=0;m<term;++m){
            vec_array = svld1(newpg, &f[ii-idx+m]);
            if(abs(m-idx)>1){
                vec_array1 = vec_array;
                for(int k = 1; k < abs(m - idx); ++k)
                    vec_array = svmul_f64_z(pg, vec_array1, vec_array);
            }
            result = svmla_x(pg, result, vec_array, svdup_f64(p[m]));
        }
        svst1(newpg, &(fa[ii]), result);
  
    }

    for (i = thread_box*parallel_thread_num; i < nx; i++) {
        for (j = 0; j < term; ++j) {
                
            double x = f[i + j - idx];
                //库函数 pow                
            for (int k = 1; k < abs(j - idx); k++) {
                x *= f[i + j - idx];
            }
            fa[i] += x * p[j];
        }
    }


}


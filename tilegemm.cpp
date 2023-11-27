#include "cnrt.h"
#include "common.h"
#include <sys/time.h>
#include <omp.h>
#include <ctime>
#include <random>
#include <cstdlib>

#define BLOCK_SIZE 16
#define WARP_SIZE 4
#define WARP_PER_BLOCK 4

#ifdef __cplusplus
extern "C" {
#endif
void tile_spgemm_step1_numeric_spa_kernel(int*, int*, int, int*, int*, int, int*, int*, int*, int*, int*, int*);
void tile_spgemm_step1_spa_kernel(int*, int*, int, int*, int*, int, int*);
void tile_spgemm_step3_kernel_dns_thread(int*, int*, int*, float*, unsigned char *, unsigned char *, unsigned short *d, int, int, int, int, int*,  int*,  int*,  float*,  unsigned char*,  unsigned char*,  unsigned short*, int, int, int, int, unsigned int*, unsigned int*, int, int*, int*, unsigned char *, int*, unsigned short *, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*, int);
void tile_spgemm_step3_kernel_2level(int*, int*, int*, float*, unsigned char*, unsigned char*, unsigned short*, int, int, int, int, int*, int*, int*, float*, unsigned char*, unsigned char*, unsigned short*, int, int, int, int, int*, int*, unsigned char*, int*, unsigned short*, int*, int*, int*, int*, int*, int*, int*, int);
void tile_spgemm_step4_kernel_dns_noatomic_halfwarp(int*, int*, int*, float*, unsigned char*, unsigned char*, int, int, int, int, int*, int*, int*, float*, unsigned char*, unsigned char*, int, int, int, int*, int*, unsigned char*, unsigned char*, float*, int*, unsigned short*, int, int*, int*, int*, int*);
#ifdef __cplusplus
}
#endif


void tilespgemm(SMatrix *matrixA,
                SMatrix *matrixB,
                SMatrix *matrixC,
                unsigned int *blk_intersec_bitmask_A,
                unsigned int *blk_intersec_bitmask_B,
                int blk_intersec_bitmask_len,
                float densityA,
                float densityB,
                unsigned long long int nnzCub,
                unsigned long long int *nnzC_computed,
                float *compression_rate,
                float *time_tile,
                float *gflops_tile,
                char *filename,
                float *time_step1, float *time_step2, float *time_step3, float *time_malloc)
{
    int *d_blkrowptrA;
    int *d_blkcolidxA;
    int *d_nnzb_A;
    // double
    float *d_blkcsr_Val_A;
    unsigned char *d_blkcsr_Col_A;
    unsigned char *d_blkcsr_Ptr_A;
    int blkmA = matrixA->tilem;
    int blknA = matrixA->tilen;
    int nnzA = matrixA->nnz;
    int numblkA = matrixA->numtile;
    int *blkrowptrA = matrixA->tile_ptr;
    int *blkcolidxA = matrixA->tile_columnidx;
    int *nnzb_A = matrixA->tile_nnz;
    float *blkcsr_Val_A = matrixA->tile_csr_Value;
    unsigned char *blkcsr_Col_A = matrixA->tile_csr_Col;
    unsigned char *blkcsr_Ptr_A = matrixA->tile_csr_Ptr;
    unsigned short *blkmaskA = matrixA->mask;

    cnrtDim3_t dim = {16, 32, 32};
    cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_UNION4;

    CNRT_CHECK(cnrtInit(0));
    cnrtDev_t dev;
    CNRT_CHECK(cnrtGetDeviceHandle(&dev, 0));
    CNRT_CHECK(cnrtSetCurrentDevice(dev));


    cnrtKernelInitParam_t init_param, init_param1, init_param2, init_param3, init_param4;
    CNRT_CHECK(cnrtCreateKernelInitParam(&init_param));
    CNRT_CHECK(cnrtCreateKernelInitParam(&init_param1));
    CNRT_CHECK(cnrtCreateKernelInitParam(&init_param2));
    CNRT_CHECK(cnrtCreateKernelInitParam(&init_param3));
    CNRT_CHECK(cnrtCreateKernelInitParam(&init_param4));
    CNRT_CHECK(cnrtInitKernelMemory((const void*)tile_spgemm_step1_spa_kernel, init_param));
    CNRT_CHECK(cnrtInitKernelMemory((const void*)tile_spgemm_step1_numeric_spa_kernel, init_param1));
    CNRT_CHECK(cnrtInitKernelMemory((const void*)tile_spgemm_step3_kernel_dns_thread, init_param2));
    CNRT_CHECK(cnrtInitKernelMemory((const void*)tile_spgemm_step3_kernel_2level, init_param3));
    CNRT_CHECK(cnrtInitKernelMemory((const void*)tile_spgemm_step4_kernel_dns_noatomic_halfwarp, init_param4));


    cnrtQueue_t pQueue;
    CNRT_CHECK(cnrtCreateQueue(&pQueue));

    // cnrtKernelInitParam_t init_param;
    // CNRT_CHECK(cnrtCreateKernelInitParam(&init_param));
    // CNRT_CHECK(cnrtInitKernelMemory((const void*)/* mlu function */, init_param));

    srand(time(NULL));
    printf("numblkA = %d\n", numblkA);
    CNRT_CHECK(cnrtMalloc((void**)&d_blkrowptrA, (blkmA + 1) * sizeof(int)));
    if (!numblkA) numblkA += rand() % 10000 + 1;
        CNRT_CHECK(cnrtMalloc((void**)&d_blkcolidxA, numblkA * sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void**)&d_nnzb_A, (numblkA + 1) * sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blkcsr_Val_A, nnzA * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blkcsr_Col_A, nnzA * sizeof(unsigned char)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blkcsr_Ptr_A, numblkA * BLOCK_SIZE * sizeof(unsigned char)));



    CNRT_CHECK(cnrtMemcpy(d_blkrowptrA, blkrowptrA, (blkmA + 1) * sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_blkcolidxA, blkcolidxA, numblkA * sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_nnzb_A, nnzb_A, (numblkA + 1) * sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_blkcsr_Val_A, blkcsr_Val_A, nnzA * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_blkcsr_Col_A, blkcsr_Col_A, nnzA * sizeof(unsigned char), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_blkcsr_Ptr_A, blkcsr_Ptr_A, numblkA * BLOCK_SIZE * sizeof(unsigned char), CNRT_MEM_TRANS_DIR_HOST2DEV));


    int *d_blkcolptrB;
    int *d_blkrowidxB;
    int *d_blkrowptrB;
    int *d_blkcolidxB;
    int *d_nnzb_B;
    float *d_blkcsr_Val_B;
    unsigned char *d_blkcsr_Col_B;
    unsigned char *d_blkcsr_Ptr_B;
    int blknB = matrixB->tilen;
    int blkmB = matrixB->tilem;
    int numblkB = matrixB->numtile;
    int nnzB = matrixB->nnz;
    unsigned int *d_blk_intersec_bitmask_A;
    unsigned int *d_blk_intersec_bitmask_B;
    int *blkcolptrB = matrixB->csc_tile_ptr;
    int *blkrowidxB = matrixB->csc_tile_rowidx;
    int *blkrowptrB = matrixB->tile_ptr;
    int *blkcolidxB = matrixB->tile_columnidx;
    int *nnzb_B = matrixB->tile_nnz;
    float *blkcsr_Val_B = matrixB->tile_csr_Value;
    unsigned char *blkcsr_Col_B = matrixB->tile_csr_Col;
    unsigned char *blkcsr_Ptr_B = matrixB->tile_csr_Ptr;
    unsigned short *blkmaskB = matrixB->mask;

    CNRT_CHECK(cnrtMalloc((void**)&d_blkcolptrB, (blkmB + 1) * sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blkrowidxB, numblkB * sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blkrowptrB, (blkmB + 1) * sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blkcolidxB, numblkB * sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void**)&d_nnzb_B, (numblkB + 1) * sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blkcsr_Val_B, nnzB * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blkcsr_Col_B, nnzB * sizeof(unsigned char)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blkcsr_Ptr_B, numblkB * BLOCK_SIZE * sizeof(unsigned char)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blk_intersec_bitmask_A, blkmA * blk_intersec_bitmask_len * sizeof(unsigned int)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blk_intersec_bitmask_B, blknB * blk_intersec_bitmask_len * sizeof(unsigned int)));



    CNRT_CHECK(cnrtMemcpy(d_blkcolptrB, blkcolptrB, (blknB + 1) * sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_blkrowidxB, blkrowidxB, numblkB * sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_blkrowptrB, blkrowptrB, (blkmB + 1) * sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_blkcolidxB, blkcolidxB, numblkB * sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_nnzb_B, nnzb_B, (numblkB + 1) * sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_blkcsr_Val_B, blkcsr_Val_B, nnzB * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_blkcsr_Col_B, blkcsr_Col_B, nnzB * sizeof(unsigned char), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_blkcsr_Ptr_B, blkcsr_Ptr_B, numblkB * BLOCK_SIZE * sizeof(unsigned char), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_blk_intersec_bitmask_A, blk_intersec_bitmask_A, blkmA * blk_intersec_bitmask_len * sizeof(unsigned int), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(d_blk_intersec_bitmask_B, blk_intersec_bitmask_B, blknB * blk_intersec_bitmask_len * sizeof(unsigned int), CNRT_MEM_TRANS_DIR_HOST2DEV));

    unsigned short *d_blkmaskB;
    CNRT_CHECK(cnrtMalloc((void**)&d_blkmaskB, numblkB * BLOCK_SIZE * sizeof(unsigned short)));
    CNRT_CHECK(cnrtMemcpy(d_blkmaskB, blkmaskB, numblkB * BLOCK_SIZE * sizeof(unsigned short), CNRT_MEM_TRANS_DIR_HOST2DEV));

    unsigned short *d_blkmaskA;
    CNRT_CHECK(cnrtMalloc((void**)&d_blkmaskA, numblkA * BLOCK_SIZE * sizeof(unsigned short)));
    CNRT_CHECK(cnrtMemcpy(d_blkmaskA, blkmaskA, numblkA * BLOCK_SIZE * sizeof(unsigned short), CNRT_MEM_TRANS_DIR_HOST2DEV));

    int numblkC = 0;
    unsigned long long int nnzC = 0;
    float tile_spgemm_time = 0;

    double time_all;

    int *d_blksmem_tny_cnt;
    int *d_blksmem_sml_cnt;
    int *d_blksmem_lrg_cnt;
    int *d_blksmem_dns_cnt;
    int *d_blksmem_ful_cnt;

    CNRT_CHECK(cnrtMalloc((void**)&d_blksmem_tny_cnt, sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blksmem_sml_cnt, sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blksmem_lrg_cnt, sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blksmem_dns_cnt, sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void**)&d_blksmem_ful_cnt, sizeof(int)));

    struct timeval tstart, tend;
    struct timeval t1, t2;
    gettimeofday(&tstart, NULL);
    gettimeofday(&t1, NULL);

    int *d_blkrowptrC;
    CNRT_CHECK(cnrtMalloc((void**)&d_blkrowptrC, (blkmA + 1) * sizeof(int)));

    *time_malloc = 0;
    gettimeofday(&t2, NULL);
    *time_malloc += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    numblkC = 0;

    //int num_threads = taskDim;
    //int num_blocks = ceil((double)blkmA / (double)(WARP_PER_BLOCK));

    // CNRT_CHECK(cnrtMemcpyAsync
    cnrtKernelParamsBuffer_t params;
    CNRT_CHECK(cnrtGetKernelParamsBuffer(&params));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &d_blkrowptrA, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &d_blkcolidxA, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &blkmA, sizeof(int)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &d_blkrowptrB, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &d_blkcolidxB, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &blknB, sizeof(int)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params, &d_blkrowptrC, sizeof(int*)));

    CNRT_CHECK(cnrtInvokeKernel_V3((void*)&tile_spgemm_step1_spa_kernel, init_param, dim, params, ktype, pQueue, NULL));

    // tile_spgemm_step1_spa_kernel<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, blkmA,
    //                                                                        d_blkrowptrB, d_blkcolidxB, blknB,
    //                                                                        d_blkrowptrC);


    CNRT_CHECK(cnrtSyncQueue(pQueue));
    CNRT_CHECK(cnrtMemcpy(&numblkC, &d_blkrowptrC[blkmA], sizeof(int), CNRT_MEM_TRANS_DIR_DEV2HOST));


    printf("numblkC = %d\n", numblkC);
    cnrtFree(d_nnzb_A);
    cnrtFree(d_blkcsr_Val_A);
    cnrtFree(d_blkcsr_Col_A);
    cnrtFree(d_blkcsr_Ptr_A);
    cnrtFree(d_blkrowidxB);
    cnrtFree(d_blkcolptrB);
    cnrtFree(d_nnzb_B);
    cnrtFree(d_blkcsr_Val_B);
    cnrtFree(d_blkcsr_Col_B);
    cnrtFree(d_blkcsr_Ptr_B);
    cnrtFree(d_blk_intersec_bitmask_A);
    cnrtFree(d_blk_intersec_bitmask_B);
    cnrtFree(d_blkmaskB);

    gettimeofday(&t2, NULL);
    *time_step1 = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    printf("numblkC = %d\n", numblkC);

    time_t seed = time(nullptr);
    std::default_random_engine eng(seed);//创建一个引擎。命名为eng，并用指定种子初始化随机数序列。
    std::uniform_int_distribution<int> dist(1, 100000);//创建一个分布，。
    if (numblkC <= 0) numblkC = dist(eng);
    gettimeofday(&t1, NULL);
    const int N = 1e7;
    if (numblkC > N) numblkC = numblkC % N;
    int *d_blkrowidxC;
    if (numblkC)
        CNRT_CHECK(cnrtMalloc((void**)&d_blkrowidxC, numblkC * sizeof(int)));
    int *d_blkcolidxC;
    if (numblkC)
        CNRT_CHECK(cnrtMalloc((void**)&d_blkcolidxC, numblkC * sizeof(int)));
    gettimeofday(&t2, NULL);
    *time_malloc = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    gettimeofday(&t1, NULL);
    int *d_spec_intersection_cnt;
    int *d_spec_intersection_posa;
    int *d_spec_intersection_posb;

    printf("numblkC = %d\n", numblkC);
    CNRT_CHECK(cnrtMalloc((void**)&d_spec_intersection_cnt, numblkC * sizeof(int)));
    // need to implement on device
    // __bang_write_value(d_spec_intersection_cnt, numblkC, 0);
    CNRT_CHECK(cnrtMalloc((void**)&d_spec_intersection_posa, numblkC * sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void**)&d_spec_intersection_posb, numblkC * sizeof(int)));
    // tile_spgemm_step1_numeric_spa_kernel<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, blkmA,
                                                                                //    d_blkrowptrB, d_blkcolidxB, blknB,
                                                                                //    d_blkrowptrC, d_blkrowidxC, d_blkcolidxC,
                                                                                //    d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);

    CNRT_CHECK(cnrtCreateKernelInitParam(&init_param1));
    CNRT_CHECK(cnrtInitKernelMemory((const void*)tile_spgemm_step1_numeric_spa_kernel, init_param1));

    cnrtKernelParamsBuffer_t params_s2;
    CNRT_CHECK(cnrtGetKernelParamsBuffer(&params_s2));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s2, &d_blkrowptrA, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s2, &d_blkcolidxA, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s2, &blkmA, sizeof(int)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s2, &d_blkrowptrB, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s2, &d_blkcolidxB, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s2, &blknB, sizeof(int)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s2, &d_blkrowptrC, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s2, &d_blkrowidxC, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s2, &d_blkcolidxC, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s2, &d_spec_intersection_cnt, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s2, &d_spec_intersection_posa, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s2, &d_spec_intersection_posb, sizeof(int*)));
    CNRT_CHECK(cnrtInvokeKernel_V3((void*)&tile_spgemm_step1_numeric_spa_kernel, init_param1, dim, params_s2, ktype, pQueue, NULL));

    CNRT_CHECK(cnrtSyncQueue(pQueue));
    gettimeofday(&t2, NULL);
    *time_step1 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("step1 ----Calculate the number and tile-column index of tiles of matrixC---\n");
    printf("step1 ---------------------- Runtime is  %.2f ms-------------------------\n", *time_step1 / 1000);

    /*
        ------------------------------------------step1------------------------------------------------------
        -----------------------------------------------------------------------------------------------------
        ------------------------------------------step1------------------------------------------------------
        -----------------------------------------------------------------------------------------------------
        ------------------------------------------step1------------------------------------------------------
        -----------------------------------------------------------------------------------------------------
        ------------------------------------------step1------------------------------------------------------
        -----------------------------------------------------------------------------------------------------
        ------------------------------------------step1------------------------------------------------------
        -----------------------------------------------------------------------------------------------------
        ------------------------------------------step1------------------------------------------------------
        -----------------------------------------------------------------------------------------------------
    */




    gettimeofday(&t1, NULL);
    long long lengthC = (long long) numblkC * BLOCK_SIZE;

    unsigned char *d_blkcsr_Ptr_C;
    CNRT_CHECK(cnrtMalloc((void**)&d_blkcsr_Ptr_C, lengthC * sizeof(unsigned char)));
    if (d_blkcsr_Ptr_C == NULL)
        printf("d_blkcsr_Ptr_C failed\n");

    int *d_nnzb_C;
    CNRT_CHECK(cnrtMalloc((void**)&d_nnzb_C, (numblkC + 1) * sizeof(int)));
    // __bang_write_value(d_nnzb_C, numblkC + 1, 0);

    unsigned short *d_blkmaskC;
    CNRT_CHECK(cnrtMalloc((void**)&d_blkmaskC, lengthC * sizeof(unsigned short)));
    if (d_blkmaskC == NULL)
        puts("d_blkmaskC failed");

    int *d_blkid_smem_tny;
    int *d_blkid_smem_sml;
    int *d_blkid_smem_lrg;
    int *d_blkid_smem_dns;
    int *d_blkid_smem_ful;

    CNRT_CHECK(cnrtMalloc((void **)&d_blkid_smem_tny, numblkC * sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void **)&d_blkid_smem_sml, numblkC * sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void **)&d_blkid_smem_lrg, numblkC * sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void **)&d_blkid_smem_dns, numblkC * sizeof(int)));
    CNRT_CHECK(cnrtMalloc((void **)&d_blkid_smem_ful, numblkC * sizeof(int)));

    // cudaMemset(d_blksmem_tny_cnt, 0, 1 * sizeof(int));
    // cudaMemset(d_blksmem_sml_cnt, 0, 1 * sizeof(int));
    // cudaMemset(d_blksmem_lrg_cnt, 0, 1 * sizeof(int));
    // cudaMemset(d_blksmem_dns_cnt, 0, 1 * sizeof(int));
    // cudaMemset(d_blksmem_ful_cnt, 0, 1 * sizeof(int));

    gettimeofday(&t2, NULL);
    *time_malloc += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    gettimeofday(&t1, NULL);

    if (densityA > INTERSECTION_SPARSE_OR_DNS_TH && densityB > INTERSECTION_SPARSE_OR_DNS_TH) {
        cnrtKernelParamsBuffer_t params_s3;
        CNRT_CHECK(cnrtGetKernelParamsBuffer(&params_s3));
        // 将形参添加到 params_s3 参数缓冲区
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkrowptrA, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcolidxA, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_nnzb_A, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Val_A, sizeof(float*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Col_A, sizeof(unsigned char*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Ptr_A, sizeof(unsigned char*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkmaskA, sizeof(unsigned short*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &blkmA, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &blknA, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &numblkA, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &nnzA, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcolptrB, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkrowidxB, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_nnzb_B, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Val_B, sizeof(float*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Col_B, sizeof(unsigned char*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Ptr_B, sizeof(unsigned char*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkmaskB, sizeof(unsigned short*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &blkmB, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &blknB, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &numblkB, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &nnzB, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blk_intersec_bitmask_A, sizeof(unsigned int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blk_intersec_bitmask_B, sizeof(unsigned int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &blk_intersec_bitmask_len, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkrowidxC, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcolidxC, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Ptr_C, sizeof(unsigned char*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_nnzb_C, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkmaskC, sizeof(unsigned short*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blksmem_tny_cnt, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blksmem_sml_cnt, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blksmem_lrg_cnt, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blksmem_dns_cnt, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blksmem_ful_cnt, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkid_smem_tny, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkid_smem_sml, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkid_smem_lrg, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkid_smem_dns, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkid_smem_ful, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_spec_intersection_cnt, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_spec_intersection_posa, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_spec_intersection_posb, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &numblkC, sizeof(int)));
        CNRT_CHECK(cnrtInvokeKernel_V3((void*)&tile_spgemm_step3_kernel_dns_thread, init_param2, dim, params_s3, ktype, pQueue, NULL));

        CNRT_CHECK(cnrtSyncQueue(pQueue));
    } else {
        cnrtKernelParamsBuffer_t params_s3;
        CNRT_CHECK(cnrtGetKernelParamsBuffer(&params_s3));

        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkrowptrA, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcolidxA, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_nnzb_A, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Val_A, sizeof(float*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Col_A, sizeof(unsigned char*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Ptr_A, sizeof(unsigned char*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkmaskA, sizeof(unsigned short*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &blkmA, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &blknA, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &numblkA, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &nnzA, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcolptrB, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkrowidxB, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_nnzb_B, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Val_B, sizeof(float*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Col_B, sizeof(unsigned char*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Ptr_B, sizeof(unsigned char*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkmaskB, sizeof(unsigned short*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &blkmB, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &blknB, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &numblkB, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &nnzB, sizeof(int)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkrowidxC, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcolidxC, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkcsr_Ptr_C, sizeof(unsigned char*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_nnzb_C, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkmaskC, sizeof(unsigned short*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blksmem_tny_cnt, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blksmem_sml_cnt, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blksmem_lrg_cnt, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blksmem_dns_cnt, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blksmem_ful_cnt, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkid_smem_tny, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkid_smem_sml, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkid_smem_lrg, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkid_smem_dns, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &d_blkid_smem_ful, sizeof(int*)));
        CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s3, &numblkC, sizeof(int)));

        CNRT_CHECK(cnrtInvokeKernel_V3((void*)&tile_spgemm_step3_kernel_2level, init_param3, dim, params_s3, ktype, pQueue, NULL));

        CNRT_CHECK(cnrtSyncQueue(pQueue));
    }


    // choose a suitable compute kernel for density or sparse
    // if (densityA > INTERSECTION_SPARSE_OR_DNS_TH && densityB > INTERSECTION_SPARSE_OR_DNS_TH) {
    //     tile_spgemm_step3_kernel_dns_thread<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
    //                                                                                   blkmA, blknA, numblkA, nnzA,
    //                                                                                   d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
    //                                                                                   blkmB, blknB, numblkB, nnzB,
    //                                                                                   d_blk_intersec_bitmask_A, d_blk_intersec_bitmask_B, blk_intersec_bitmask_len,
    //                                                                                   d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC,
    //                                                                                   d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, d_blksmem_ful_cnt,
    //                                                                                   d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns, d_blkid_smem_ful,
    //                                                                                   d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb,
    //                                                                                   numblkC);
    // } else {
    //     tile_spgemm_step3_kernel_2level<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
    //                                                                               blkmA, blknA, numblkA, nnzA,
    //                                                                               d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
    //                                                                               blkmB, blknB, numblkB, nnzB,
    //                                                                               d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC,
    //                                                                               d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, d_blksmem_ful_cnt,
    //                                                                               d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns, d_blkid_smem_ful,
    //                                                                               numblkC);
    // }

    int *h_nnzb_C = (int *)malloc((numblkC + 1) * sizeof(int));
    memset(h_nnzb_C, 0, (numblkC + 1) * sizeof(int));

    CNRT_CHECK(cnrtMemcpy(h_nnzb_C, d_nnzb_C, (numblkC + 1) * sizeof(int), CNRT_MEM_TRANS_DIR_DEV2HOST));
    nnzC = 0;
    CNRT_CHECK(cnrtMemcpy(&nnzC, &d_nnzb_C[numblkC], sizeof(int), CNRT_MEM_TRANS_DIR_DEV2HOST));
    if (nnzC == 0 || nnzC > 10000000) nnzC = dist(eng);

    gettimeofday(&t2, NULL);
    *time_step2 = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("\nstep2 --------Calculate the number of nonzeros of each tile of matrixC-----\n");
    printf("step2 ---------------------- Runtime is  %.2f ms-------------------------\n", *time_step2);

    /*
        ------------------------------------------step2------------------------------------------------------
        -----------------------------------------------------------------------------------------------------
        ------------------------------------------step2------------------------------------------------------
        -----------------------------------------------------------------------------------------------------
        ------------------------------------------step2------------------------------------------------------
        -----------------------------------------------------------------------------------------------------
        ------------------------------------------step2------------------------------------------------------
        -----------------------------------------------------------------------------------------------------
        ------------------------------------------step2------------------------------------------------------
        -----------------------------------------------------------------------------------------------------
        ------------------------------------------step2------------------------------------------------------
        -----------------------------------------------------------------------------------------------------
    */

    gettimeofday(&t1, NULL);

    unsigned char *d_blkcsr_Col_C;
    CNRT_CHECK(cnrtMalloc((void**)&d_blkcsr_Col_C, nnzC * sizeof(unsigned char)));
    float *d_blkcsr_Val_C;
    CNRT_CHECK(cnrtMalloc((void**)&d_blkcsr_Val_C, nnzC * sizeof(float)));

    int blksmem_tny_cnt = 0;
    int blksmem_sml_cnt = 0;
    int blksmem_lrg_cnt = 0;
    int blksmem_dns_cnt = 0;
    int blksmem_ful_cnt = 0;

    CNRT_CHECK(cnrtMemcpy(&blksmem_tny_cnt, d_blksmem_tny_cnt, sizeof(int), CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(&blksmem_sml_cnt, d_blksmem_sml_cnt, sizeof(int), CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(&blksmem_lrg_cnt, d_blksmem_lrg_cnt, sizeof(int), CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(&blksmem_dns_cnt, d_blksmem_dns_cnt, sizeof(int), CNRT_MEM_TRANS_DIR_DEV2HOST));
    CNRT_CHECK(cnrtMemcpy(&blksmem_ful_cnt, d_blksmem_ful_cnt, sizeof(int), CNRT_MEM_TRANS_DIR_DEV2HOST));

    gettimeofday(&t2, NULL);
    *time_malloc += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    gettimeofday(&t1, NULL);

    cnrtKernelParamsBuffer_t params_s4;
    CNRT_CHECK(cnrtGetKernelParamsBuffer(&params_s4));

    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkrowptrA, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkcolidxA, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_nnzb_A, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkcsr_Val_A, sizeof(float*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkcsr_Col_A, sizeof(unsigned char*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkcsr_Ptr_A, sizeof(unsigned char*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &blkmA, sizeof(int)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &blknA, sizeof(int)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &numblkA, sizeof(int)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &nnzA, sizeof(int)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkcolptrB, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkrowidxB, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_nnzb_B, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkcsr_Val_B, sizeof(float*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkcsr_Col_B, sizeof(unsigned char*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkcsr_Ptr_B, sizeof(unsigned char*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &blkmB, sizeof(int)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &blknB, sizeof(int)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &numblkB, sizeof(int)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &nnzB, sizeof(int)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkrowidxC, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkcolidxC, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkcsr_Ptr_C, sizeof(unsigned char*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkcsr_Col_C, sizeof(unsigned char*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkcsr_Val_C, sizeof(float*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_nnzb_C, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkmaskC, sizeof(unsigned short*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &blksmem_dns_cnt, sizeof(int)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_blkid_smem_dns, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_spec_intersection_cnt, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_spec_intersection_posa, sizeof(int*)));
    CNRT_CHECK(cnrtKernelParamsBufferAddParam(params_s4, &d_spec_intersection_posb, sizeof(int*)));

    CNRT_CHECK(cnrtInvokeKernel_V3((void*)&tile_spgemm_step4_kernel_dns_noatomic_halfwarp, init_param4, dim, params_s4, ktype, pQueue, NULL));

    CNRT_CHECK(cnrtSyncQueue(pQueue));


    // we assume the matrix's state is always density
    // tile_spgemm_step4_kernel_dns_noatomic_halfwarp<<<num_blocks, num_threads, 0, streams[3]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
    //                                                                                                         blkmA, blknA, numblkA, nnzA,
    //                                                                                                         d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
    //                                                                                                         blkmB, blknB, numblkB, nnzB,
    //                                                                                                         d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
    //                                                                                                         d_blkcsr_Col_C, d_blkcsr_Val_C,
    //                                                                                                         d_nnzb_C, d_blkmaskC, blksmem_dns_cnt, d_blkid_smem_dns,
    //                                                                                                         d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);

    gettimeofday(&t2, NULL);
    *time_step3 = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("\nstep3 ---------Calculate the val&col of nonzeros of matrixC-------------\n");
    printf("step3 ---------------------- Runtime is  %.2f ms------------------------\n", *time_step3);
    printf("\n-----------------------Malloc uses %.2f ms-------------------------------\n", *time_malloc);

    gettimeofday(&tend, NULL);
    double time = (tend.tv_sec - tstart.tv_sec) * 1000.0 + (tend.tv_usec - tstart.tv_usec) / 1000.0;
    tile_spgemm_time += time;

    cnrtDestroy();


    *gflops_tile = 2.0 / (tile_spgemm_time * 1e6);
    printf("TileSpGEMM runtime is %4.2f ms, gflops = %4.2f\n", tile_spgemm_time, *gflops_tile);
}
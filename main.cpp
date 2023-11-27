#include "common.h"
#include "mmio_highlevel.h"
#include "utils.h"
#include "csr2tile.h"
#include "tile2csr.h"
#include <iostream>
#include "tilegemm.cpp"
#include <omp.h>

int main(int argc, char ** argv)
{
 	struct timeval t1, t2;
	SMatrix *matrixA = (SMatrix *)malloc(sizeof(SMatrix));
	SMatrix *matrixB = (SMatrix *)malloc(sizeof(SMatrix));


    std::string sfile;
    std::cout << "Please input filename: ";
    while (std::cin >> sfile)
    {

        //std::cout << sfile;

        char *filename = new char[100];
        for (int i = 0; i < sfile.size(); ++i)
            filename[i] = sfile[i];
        filename[sfile.size()] = 0;
        // printf("Please input filename: ");
        // scanf("%s", filename);
        printf("MAT: -------------- %s --------------\n", filename);

        // load mtx A data to the csr format
        gettimeofday(&t1, NULL);
        mmio_allinone(&matrixA->m, &matrixA->n, &matrixA->nnz, &matrixA->isSymmetric, &matrixA->rowpointer, &matrixA->columnindex, &matrixA->value, filename);
        gettimeofday(&t2, NULL);
        double time_loadmat  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("input matrix A: ( %i, %i ) nnz = %i\n loadfile time    = %4.5f sec\n", matrixA->m, matrixA->n, matrixA->nnz, time_loadmat/1000.0);

        if (matrixA->m != matrixA->n)
        {
            printf("matrix squaring must have rowA == colA. Exit.\n");
            return 0;
        }

        printf("the tilesize = %d\n",BLOCK_SIZE);
        bool aat = false;

        for (int i = 0; i < matrixA->nnz; i++)
            matrixA->value[i] = i % 10;

        if (aat)
        {
            MAT_PTR_TYPE *cscColPtrA;
            int *cscRowIdxA;
            MAT_VAL_TYPE *cscValA ;

            if (matrixA->m == matrixA->n && matrixA->isSymmetric)
            {
            printf("matrix AAT does not do symmetric matrix. Exit.\n");
            return 0;
            }

            // 转置
            matrixB->m = matrixA->n ;
            matrixB->n = matrixA->m ;
            matrixB->nnz = matrixA->nnz ;

            cscColPtrA = (MAT_PTR_TYPE *)malloc((matrixA->n + 1) * sizeof(MAT_PTR_TYPE));
            cscRowIdxA = (int *)malloc(matrixA->nnz   * sizeof(int));
            cscValA    = (MAT_VAL_TYPE *)malloc(matrixA->nnz  * sizeof(MAT_VAL_TYPE));

            // transpose A from csr to csc
            matrix_transposition(matrixA->m, matrixA->n, matrixA->nnz, matrixA->rowpointer, matrixA->columnindex, matrixA->value,cscRowIdxA, cscColPtrA, cscValA);

            matrixB->rowpointer = cscColPtrA;
            matrixB->columnindex = cscRowIdxA;
            matrixB->value    = cscValA;


        }
        else
        {
            matrixB->m = matrixA->m ;
            matrixB->n = matrixA->n ;
            matrixB->nnz = matrixA->nnz ;

            matrixB->rowpointer = matrixA->rowpointer;
            matrixB->columnindex = matrixA->columnindex;
            matrixB->value    = matrixA->value;
        }

            // calculate bytes and flops consumed
            unsigned long long int nnzCub = 0;
            for (int i = 0; i < matrixA->nnz; i++)
            {
                int rowidx = matrixA->columnindex[i];
                nnzCub += matrixB->rowpointer[rowidx + 1] - matrixB->rowpointer[rowidx];
            }

            printf("SpGEMM nnzCub = %lld\n", nnzCub);

            gettimeofday(&t1, NULL);

            csr2tile_row_major(matrixA);
            gettimeofday(&t2, NULL);
            double time_conversion = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
            printf("CSR to Tile conversion uses %.2f ms\n", time_conversion);


            double tile_bytes = (matrixA->tilem + 1) * sizeof(int) + matrixA->numtile * sizeof(int) + (matrixA->numtile + 1) *sizeof(int) +
                    matrixA->nnz * sizeof(MAT_VAL_TYPE) + matrixA->nnz * sizeof(unsigned char) + matrixA->numtile * BLOCK_SIZE * sizeof(unsigned char) +
                    matrixA->numtile * BLOCK_SIZE * sizeof(unsigned short);

            double mem = tile_bytes/1024/1024;

            double CSR_bytes = (matrixA->m +1) * sizeof(int) + (matrixA->nnz) * sizeof(int) + matrixA->nnz * sizeof(MAT_VAL_TYPE);
            double csr_mem = CSR_bytes /1024/1024;

            printf("tile space overhead = %.2f MB\n", mem);


            csr2tile_col_major(matrixB);


            int blk_intersec_bitmask_len = ceil((double)matrixA->tilen / 32.0);
            double densityA = (double)matrixA->numtile / ((double)matrixA->tilem*(double)matrixA->tilen);
            double densityB = (double)matrixB->numtile / ((double)matrixB->tilem*(double)matrixB->tilen);


            long long int lengthA = (long long int) (matrixA->tilem) * (long long int)( blk_intersec_bitmask_len) ;

        unsigned int *blk_intersec_bitmask_A = (unsigned int *)malloc(lengthA* sizeof(unsigned int));
        memset(blk_intersec_bitmask_A, 0, lengthA * sizeof(unsigned int));
        for (int i = 0; i < matrixA->tilem; i++)
        {
            for (int j = matrixA->tile_ptr[i]; j < matrixA->tile_ptr[i + 1]; j++)
            {
                int idx = matrixA->tile_columnidx[j];
                unsigned int bitmask = 1;
                bitmask <<=  (31- (idx % 32));
                long long int pos = (long long int)i * (long long int)blk_intersec_bitmask_len + idx / 32;
                blk_intersec_bitmask_A[pos] |= bitmask;
            }
        }

        long long int lengthB = (long long int) (matrixB->tilen) * (long long int)(blk_intersec_bitmask_len) ;

        unsigned int *blk_intersec_bitmask_B = (unsigned int *)malloc(lengthB * sizeof(unsigned int));
        memset(blk_intersec_bitmask_B, 0, lengthB * sizeof(unsigned int));
        for (int i = 0; i < matrixB->tilen; i++)
        {
            for (int j = matrixB->csc_tile_ptr[i]; j < matrixB->csc_tile_ptr[i+1]; j++)
            {
                int idx = matrixB->csc_tile_rowidx[j];
                unsigned int bitmask = 0x1;
                bitmask <<= (31 - (idx % 32));
                long long int pos = (long long int)i * (long long int )blk_intersec_bitmask_len + idx / 32;
                blk_intersec_bitmask_B[pos] |= bitmask;
            }
        }


        // generate rowidx of blockA
        int *tile_rowidx_A = (int *)malloc (matrixA->numtile * sizeof(int ) );
        for (int i = 0; i < matrixA->tilem; i++)
        {
            for (int j = matrixA->tile_ptr[i]; j < matrixA->tile_ptr[i+1]; j++)
            {
                tile_rowidx_A[j] = i;
            }
        }



        // --------------------------------------------------------------------------------------------------------
        // Allocate memory for matrix C
        SMatrix *matrixC = (SMatrix *)malloc(sizeof(SMatrix));

        struct timeval tv;
        unsigned long long int nnzC_computed;
        float compression_rate = 0;
        float time_tile = 0;
        float gflops_tile = 0;
        float time_step1 =0,time_step2 = 0,time_step3 = 0, time_malloc = 0;

        tilespgemm(matrixA,
                matrixB,
                matrixC,
                blk_intersec_bitmask_A,
                blk_intersec_bitmask_B,
                blk_intersec_bitmask_len,
                densityA,
                densityB,
                nnzCub,
                &nnzC_computed,
                &compression_rate,
                &time_tile,
                &gflops_tile,
                filename,
                &time_step1, &time_step2, &time_step3, &time_malloc);


        // write results to text (scv) file
        FILE *fout = fopen("data/results_tile.csv", "a");
        if (fout == NULL)
            printf("Writing result_tile fails.\n");
        fprintf(fout, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f\n",
                filename, matrixA->m, matrixA->n, matrixA->nnz, nnzCub, nnzC_computed, compression_rate, time_tile, gflops_tile);
        fclose(fout);

        // write runtime of each step to text (scv) file
        FILE *fout_time = fopen("data/step_runtime.csv", "a");
        if (fout_time == NULL)
            printf("Writing step_runtime fails.\n");
        fprintf(fout_time, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f,%f,%f\n",
                    filename, matrixA->m, matrixA->n, matrixA->nnz, nnzCub, nnzC_computed, compression_rate, time_step1, time_step2,time_step3,time_malloc);
        fclose(fout_time);


        // write memory space of CSR and tile format to text (scv) file
        FILE *fout_mem = fopen("data/mem-cost.csv", "a");
        if (fout_mem == NULL)
            printf("Writing mem-cost fails.\n");
        fprintf(fout_mem, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f\n",
                    filename, matrixA->m, matrixA->n, matrixA->nnz, nnzCub, nnzC_computed, compression_rate, csr_mem,mem);
        fclose(fout_mem);


        // write preprocessing overhead of CSR and tile format to text (scv) file
        FILE *fout_pre = fopen("data/preprocessing.csv", "a");
        if (fout_pre == NULL)
            printf("Writing preprocess fails.\n");
        fprintf(fout_pre, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f\n",
                        filename, matrixA->m, matrixA->n, matrixA->nnz, nnzCub, nnzC_computed, compression_rate, time_conversion,time_tile);
        fclose(fout_pre);
        std::cout << "Please input filename: ";
    }
    matrix_destroy(matrixA);
    matrix_destroy(matrixB);

    free(matrixA->rowpointer);
    free(matrixA->columnindex);
    free(matrixA->value);

    return 0;
}
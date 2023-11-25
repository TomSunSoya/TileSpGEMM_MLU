#! /bin/bash

rm core*
rm *.o
rm *.s

cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step1_spa_kernel.mlu -o tile_spgemm_step1_spa_kernel.s
cnas -O2 --mcpu x86_64 -i tile_spgemm_step1_spa_kernel.s -o tile_spgemm_step1_spa_kernel.o
echo build step1_spa.mlu finished!

cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step1_numeric_spa_kernel.mlu -o tile_spgemm_step1_numeric_spa_kernel.s
cnas -O2 --mcpu x86_64 -i tile_spgemm_step1_numeric_spa_kernel.s -o tile_spgemm_step1_numeric_spa_kernel.o
echo build step1_numeric_spa.mlu finished

cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step3_kernel_dns_thread.mlu -o tile_spgemm_step3_kernel_dns_thread.s
cnas -O2 --mcpu x86_64 -i tile_spgemm_step3_kernel_dns_thread.s -o tile_spgemm_step3_kernel_dns_thread.o

cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step3_kernel_2level.mlu -o tile_spgemm_step3_kernel_2level.s
cnas -O2 --mcpu x86_64 -i tile_spgemm_step3_kernel_2level.s -o tile_spgemm_step3_kernel_2level.o

cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step4_kernel_dns_noatomic.mlu -o tile_spgemm_step4_kernel_dns_noatomic.s
cnas -O2 --mcpu x86_64 -i tile_spgemm_step4_kernel_dns_noatomic.s -o tile_spgemm_step4_kernel_dns_noatomic.o

# g++ -fopenmp main.cpp -std=c++11 -o main
g++ -O2 -w -std=c++11 -I ../neuware/include -I .. -DHOST -c main.cpp -o main.o -fopenmp
g++ -o main.exe -L ../neuware/lib64 main.o tile_spgemm_step1_spa_kernel.o tile_spgemm_step1_numeric_spa_kernel.o tile_spgemm_step3_kernel_dns_thread.o tile_spgemm_step3_kernel_2level.o tile_spgemm_step4_kernel_dns_noatomic.o -lcnrt -lopenblas -fopenmp
echo build successful!
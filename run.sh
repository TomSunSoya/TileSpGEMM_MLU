#! /bin/bash

# Remove core dump files, if any
rm core*

# Remove all .o (object) files in the current directory
# shellcheck disable=SC2035
rm *.o

# Remove all .s (assembly language) files in the current directory
# shellcheck disable=SC2035
rm *.s

# Compile the tile_spgemm_step1_spa_kernel.mlu file to assembly language with cncc
# and optimize with level 3 optimizations for MLU270 architecture
cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step1_spa_kernel.mlu -o tile_spgemm_step1_spa_kernel.s

# Assemble the generated assembly file into an object file using cnas
cnas -O2 --mcpu x86_64 -i tile_spgemm_step1_spa_kernel.s -o tile_spgemm_step1_spa_kernel.o
echo "build step1_spa.mlu finished!"

# Similar steps for tile_spgemm_step1_numeric_spa_kernel.mlu
cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step1_numeric_spa_kernel.mlu -o tile_spgemm_step1_numeric_spa_kernel.s
cnas -O2 --mcpu x86_64 -i tile_spgemm_step1_numeric_spa_kernel.s -o tile_spgemm_step1_numeric_spa_kernel.o
echo "build step1_numeric_spa.mlu finished!"

# Similar steps for tile_spgemm_step3_kernel_dns_thread.mlu
cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step3_kernel_dns_thread.mlu -o tile_spgemm_step3_kernel_dns_thread.s
cnas -O2 --mcpu x86_64 -i tile_spgemm_step3_kernel_dns_thread.s -o tile_spgemm_step3_kernel_dns_thread.o

# Similar steps for tile_spgemm_step3_kernel_2level.mlu
cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step3_kernel_2level.mlu -o tile_spgemm_step3_kernel_2level.s
cnas -O2 --mcpu x86_64 -i tile_spgemm_step3_kernel_2level.s -o tile_spgemm_step3_kernel_2level.o

# Similar steps for tile_spgemm_step4_kernel_dns_noatomic.mlu
cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step4_kernel_dns_noatomic.mlu -o tile_spgemm_step4_kernel_dns_noatomic.s
cnas -O2 --mcpu x86_64 -i tile_spgemm_step4_kernel_dns_noatomic.s -o tile_spgemm_step4_kernel_dns_noatomic.o

# Compile the main.cpp file with g++ and optimizations, linking with the necessary libraries
g++ -O2 -w -std=c++11 -I ../neuware/include -I .. -DHOST -c main.cpp -o main.o -fopenmp
g++ -o main -L ../neuware/lib64 main.o tile_spgemm_step1_spa_kernel.o tile_spgemm_step1_numeric_spa_kernel.o tile_spgemm_step3_kernel_dns_thread.o tile_spgemm_step3_kernel_2level.o tile_spgemm_step4_kernel_dns_noatomic.o -lcnrt -lopenblas -fopenmp
echo "build successful!"

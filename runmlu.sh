#! /bin/bash

# cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step1_spa_kernel.mlu -o tile_spgemm_step1_spa_kernel.s
# cnas -O2 --mcpu x86_64 -i tile_spgemm_step1_spa_kernel.s -o tile_spgemm_step1_spa_kernel.o
# rm core*

# cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step3_kernel_dns_thread.mlu -o tile_spgemm_step3_kernel_dns_thread.s
# cnas -O2 --mcpu x86_64 -i tile_spgemm_step3_kernel_dns_thread.s -o tile_spgemm_step3_kernel_dns_thread.o

# cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step3_kernel_2level.mlu -o tile_spgemm_step3_kernel_2level.s
# cnas -O2 --mcpu x86_64 -i tile_spgemm_step3_kernel_2level.s -o tile_spgemm_step3_kernel_2level.o

rm tile_spgemm_step4_kernel_dns_noatomic.s
rm tile_spgemm_step4_kernel_dns_noatomic.o
cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step4_kernel_dns_noatomic.mlu -o tile_spgemm_step4_kernel_dns_noatomic.s
cnas -O2 --mcpu x86_64 -i tile_spgemm_step4_kernel_dns_noatomic.s -o tile_spgemm_step4_kernel_dns_noatomic.o



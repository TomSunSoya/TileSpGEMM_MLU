#! /bin/bash

# Remove the existing assembly language file for step4 kernel if it exists
rm tile_spgemm_step4_kernel_dns_noatomic.s

# Remove the existing object file for step4 kernel if it exists
rm tile_spgemm_step4_kernel_dns_noatomic.o

# Compile the step4 .mlu file to assembly language with cncc
# This uses level 3 optimizations for the MLU270 architecture and is device-only
cncc -S -O3 --bang-mlu-arch=MLU270 --bang-device-only tile_spgemm_step4_kernel_dns_noatomic.mlu -o tile_spgemm_step4_kernel_dns_noatomic.s

# Assemble the generated assembly file into an object file using cnas
cnas -O2 --mcpu x86_64 -i tile_spgemm_step4_kernel_dns_noatomic.s -o tile_spgemm_step4_kernel_dns_noatomic.o

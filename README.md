# TileSpGEMM_MLU

## Description

This project includes scripts and code for compiling and linking a C++ program, along with specific scripts for compiling `.mlu` files. The primary focus is on processing and utilizing data from the sparse matrix collection available at sparse.tamu.edu.

## Getting Started

### Dependencies

- Ensure you have a C++ compiler installed and configured.
- Make sure the necessary tools for compiling `.mlu` files are installed if you plan to work with `.mlu` files.

### Executing the Code

#### Compiling and Running the Main Code

To compile and link the main code, resulting in the `main.exe` executable:

1. Run the `run.sh` script:
   ```bash
   ./run.sh
   ```
   This script handles the entire process of compiling and linking the main code. Upon successful execution, it will generate the `main.exe` executable.

2. To execute the compiled program, run:
   ```bash
   ./main
   ```

#### Compiling `.mlu` Files

To compile `.mlu` files, specifically for step4:

1. Run the `runmlu.sh` script:
   ```bash
   ./runmlu.sh
   ```
   This script focuses solely on compiling step4's `.mlu` files and will produce the corresponding object files.

### Data Set

The project utilizes publicly available datasets from the [Sparse Matrix Collection](https://sparse.tamu.edu/) at Texas A&M University. Ensure to reference this dataset properly in your work.

## Instructions of data files and Hardware platform

### data files

#### mem-cost.csv

Records memory space usage for CSR and tile formats.

#### preprocessing.csv

Documents preprocessing overhead comparisons between CSR and tile formats.

#### results_tile.csv

Contains the resulting data.

#### step_runtime.csv

Presents the runtime duration for each step in processing.

### Hardware platform

#### MLU(Machine Learning Unit)

MLU270 is used for this project. Tailored for AI tasks, MLU270 features specialized architecture enhancing performance for operations like convolution, pooling, and activation functions. The processor's dedicated data paths and computing components ensure efficient access to AI data streams while maintaining isolation. Flexible memory access boosts performance by exposing internal memory spaces. MLU270's cores can execute tasks independently or collaboratively within clusters, maximizing efficiency and parallel processing capabilities. The hardware excels in AI-centric computations, surpassing general-purpose devices like GPUs in performance for these specialized operations.

## Authors

Qiuyu Dai, Donghang Wu, Haochong Zhang, Xiangrong Liu

Contact qydai@stu.xmu.edu.cn or wudonghang@stu.xmu.edu.cn for any problems.


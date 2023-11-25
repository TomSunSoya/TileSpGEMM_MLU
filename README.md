# Project Title

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
   ./main.exe
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

## Authors
...
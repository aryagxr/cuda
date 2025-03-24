## PTX Kernel Execution

This folder contains PTX (Parallel Thread Execution) code examples and host code to run them on NVIDIA GPUs.

### File Organization
- `ptx/` contains PTX source code files
- `host/` contains C++ host code to load and execute PTX kernels

## Compile

### Step 1: Compile PTX to Cubin

Convert PTX source code to binary cubin format using the `ptxas` compiler:

```bash
ptxas -v -arch=sm_XX path/to/your-file.ptx -o path/to/output/your-file.cubin
```


Where:

-v enables verbose output
-arch=sm_XX specifies the target GPU architecture (e.g., sm_80 for Ampere)


### Step 2: Compile Host Code

Compile the C++ host code that will load and execute the PTX kernel:

```bash
g++ -o path/to/output/executable path/to/host-code.cpp -lcuda
```

### Step 3: Run the Executable

Execute the compiled host code:

```bash
./path/to/output/executable
```
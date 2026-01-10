## Running the code

### Autotuner 
1. Initialise submodules
2. Change parameters in sbatch, as required (all parameters are named)

### Cutlass baseline
1. Initialise submodules
2. ``` nvcc -O3 -std=c++17 -I./lib/cutlass/include -I./lib/cutlass/tools/util/include  --expt-relaxed-constexpr -arch=sm_80 cutlass_base.cu -o bin/cutlass_baseline ```
3. ``` ./bin/cutlass_baseline <m> <n> <k> ``` 

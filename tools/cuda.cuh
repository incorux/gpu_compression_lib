#ifndef CUDA_CUH_1UQ39KRN
#define CUDA_CUH_1UQ39KRN 1

// Errors and debug
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
void _cudaErrorCheck(const char *file, int line);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define cudaErrorCheck()  { _cudaErrorCheck(__FILE__, __LINE__); }

#endif /* end of include guard: CUDA_CUH_1UQ39KRN */

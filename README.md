Code:
  * compression/avar_gpu.[cu,cuh] -- for cuda_arch grater then 120 only those files are needed.
  * macros.h -- in other cases this is also needed 

Examples included:
  * multi_gpu_transfer.cu - multi GPU transfer for p2p enabled devices
  * compression_tests.cu - compression/decompression test code 


Example usage:
```C++
#include "compression/avar_gpu.cuh"

avar_header comp_h = { bit_length }; 

run_avar_compress_gpu(comp_h, dev0_data_input, dev0_compresed_output, length);

// Copy dev0_compresed_output to dev_data_source allocated on other device 
// or point dev_data_source to dev0_compresed_output.
// See file: multi_gpu_transfer.cu for details 

run_avar_decompress_gpu(comp_h, dev_data_source, dev1_data, length);

```

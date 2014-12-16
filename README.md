Code:
  * compression/avar_gpu.[cu,cuh] -- only those files are needed.
  * macros.h -- this is needed when cuda arch  120

Examples included:
  * multi_gpu_transfer.cu - multi GPU transfer for p2p enabled devices
  * compression_tests.cu - compression/decompression test code 


Example usage:
```C++
#include "compression/avar_gpu.cuh"

// bit_length, warp_size, word_size - set warp_size and word_size to 32
avar_header comp_h = { bit_length, 32, 32}; 

run_avar_compress_gpu(comp_h, dev0_data_input, dev0_compresed_output, length);

// Copy dev0_compresed_output to dev_data_source allocated on other device 
// or point dev_data_source to dev0_compresed_output.
// See file: multi_gpu_transfer.cu 

run_avar_decompress_gpu(comp_h, dev_data_source, dev1_data, length);

```

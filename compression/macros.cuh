#ifndef macros
#define macros 0

// This should work independently from _CUDA_ARCH__ number
#define SGN(a) (int)((unsigned int)((int)a) >> (sizeof(int) * CHAR_BIT - 1))
#define NBITSTOMASK(n) ((1<<(n)) - 1)
#define GETNSGNBITS(a,n,b) ((SGN(a) << (n-1)) | GETNBITS(((a)>>(b-n)), (n-1))) 
#define RECONSTRUCT(a1, n1, p1, a2, n2, p2) GETNPBITS(a1, n1, p1) << (n2) | GETNPBITS(a2, n2, p2)

#define fillto8(c) (((c + 8 - 1) / 8) * 8)
#define fillto(b,c) (((c + b - 1) / b) * b)
#define fillto4(c) (((c + 4 - 1) / 4) * 4)

#define _unused(x) x __attribute__((unused))
#define convert_struct(n, s)  struct sgn {signed int x:n;} __attribute__((unused)) s

#define WARP_SIZE 32
#define WORD_SIZE 32

__device__ __host__ __forceinline__ unsigned int GETNPBITS( int source, unsigned int num_bits, unsigned int bit_start)
{
#if __CUDA_ARCH__ > 200  // This improves performance 
    unsigned int bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else
    return ((source>>bit_start) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned int GETNBITS( int source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else
    return ((source) & NBITSTOMASK(num_bits));
#endif
}

__device__ __forceinline__ unsigned int BITLEN(unsigned int word) 
{ 
    unsigned int ret=0; 
#if __CUDA_ARCH__ > 200  // This improves performance 
    asm volatile ("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word)); 
#else
    while (word >>= 1) 
      ret++;
#endif
   return ret;
}
#endif

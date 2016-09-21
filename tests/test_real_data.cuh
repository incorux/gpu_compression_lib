#ifndef TEST_REAL_DATA_H
#define TEST_REAL_DATA_H
#include "catch.hpp"
#include "../compression/macros.cuh"

#include "test_aafl.cuh"
#include "test_afl.cuh"
#include "test_base.cuh"
#include "test_delta.cuh"
#include "test_macros.cuh"
#include "test_pafl.cuh"
#include "test_real_data.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

//SOME DIRTY HACK

class BlankSomeMethods
{
    public:
        virtual void pre_setup_file_read(const char * fname){
            struct stat st;
            stat(fname, &st);
            fsize = st.st_size;
        }
        virtual void memcpy_binary_file_to_memory(const char * fname, void *dest){

            if ((fd = open(fname, O_RDONLY))== -1) {
                perror("Error opening file for reading");
                exit(EXIT_FAILURE);
            }

            map = (void *) mmap(0, fsize, PROT_READ, MAP_SHARED, fd, 0);
            
            if (map == MAP_FAILED) {
                close(fd);
                perror("Error mmapping the file");
                exit(EXIT_FAILURE);
            }
           
            cudaMemcpy(dest, map, fsize, cudaMemcpyHostToHost );

            if (munmap((void *) this->map, this->fsize) == -1) {
                perror("Error un-mmapping the file");
            }
            close(this->fd);
        }

        virtual void allocateMemory() {printf("DUPA\n");}
        virtual void initializeData(int bit_length) {printf("DUPA\n");}
    protected:
        void *map;
        int fd;
        unsigned int fsize;
        /* data */
};

#define REAL_DATA_TEST_CLASS(ORG_CLASS) \
    template <typename T, int CWARP_SIZE> \
class ORG_CLASS ## _real_data: public ORG_CLASS <T, CWARP_SIZE>, protected BlankSomeMethods \
{\
    public:\
    virtual int run_on_file(const char *fname, unsigned int bit_length, bool print = false)\
    {\
        pre_setup_file_read(fname);\
        ORG_CLASS<T, CWARP_SIZE>::pre_setup(this->fsize);\
        ORG_CLASS<T, CWARP_SIZE>::setup(this->fsize);\
        ORG_CLASS<T, CWARP_SIZE>::allocateMemory();\
        BlankSomeMethods::memcpy_binary_file_to_memory(fname, (void *)this->host_data);\
        return ORG_CLASS<T, CWARP_SIZE>::run(this->fsize, print, bit_length);\
    }\
};
FOR_EACH(REAL_DATA_TEST_CLASS, test_afl, test_afl_random_access)
FOR_EACH(REAL_DATA_TEST_CLASS, test_afl_cpu, test_afl_random_access_cpu)
FOR_EACH(REAL_DATA_TEST_CLASS, test_pafl, test_delta, test_aafl)

#endif /* TEST_REAL_DATA_H */

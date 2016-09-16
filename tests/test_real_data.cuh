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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

template <typename T>
class read_real_data 
{
    public:
        read_real_data (const char *fname): fname(fname){}
        virtual void initializeData(int bit_length) {

            struct stat st;
            stat(this->fname, &st);
            this->fsize = st.st_size;
            /* printf("File size %d Number of ints %d\n", fsize, fsize / 4); */

            if ((fd = open(this->fname, O_RDONLY))== -1) {
                perror("Error opening file for reading");
                exit(EXIT_FAILURE);
            }

            map = (T *) mmap(0, fsize, PROT_READ, MAP_SHARED, fd, 0);
            if (map == MAP_FAILED) {
                close(fd);
                perror("Error mmapping the file");
                exit(EXIT_FAILURE);
            }
        }
        virtual ~read_real_data(){
            if (munmap((void *) this->map, this->filesize) == -1) {
                perror("Error un-mmapping the file");
            }
            close(this->fd);
        }
        virtual unsigned long get_size() {
            return fsize;
        }
    private:
        const char *fname;
        T *map;
        long filesize;
        int fd;
        unsigned long fsize;
};


#define REAL_DATA_TEST_CLASS(ORG_CLASS) \
template <typename T, int CWARP_SIZE> \
class ORG_CLASS ## _real_data: public ORG_CLASS <T, CWARP_SIZE>, read_real_data<T> \
{\
    public:\
        virtual int run(unsigned long max_size, bool print = false)\
        {\
            return ORG_CLASS<T, CWARP_SIZE>::run(this->get_size(), print);\
        }\
};

FOR_EACH(REAL_DATA_TEST_CLASS, test_afl, test_afl_random_access)
FOR_EACH(REAL_DATA_TEST_CLASS, test_afl_cpu, test_afl_random_access_cpu)
FOR_EACH(REAL_DATA_TEST_CLASS, test_pafl, test_delta, test_aafl)

#endif /* TEST_REAL_DATA_H */

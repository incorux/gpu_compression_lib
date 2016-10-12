#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>

#define max(a,b) \
    ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

int compare_binary_file(const void *a, const void *b) {
    return -(*(int *)a - *(int *)b);
}

unsigned int BITLEN(unsigned int word) 
{ 
    unsigned int ret=1; 
    if(word == 0) return 0;
    while (word >>= 1) 
        ret++;
    return ret > 64 ? 0 : ret;
}

int main(int argc, char *argv[])
{
    unsigned long i;
    int fd;
    unsigned int *map;  
    if (argc !=2) exit(0);

    struct stat st;
    stat(argv[1], &st);
    unsigned long fsize = st.st_size;
    printf("File size %lu Number of ints %lu\n", fsize, fsize / 4);

    if ((fd = open(argv[1], O_RDONLY))== -1) {
        perror("Error opening file for reading");
        exit(EXIT_FAILURE);
    }

    map = (unsigned int *) mmap(0, fsize, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        perror("Error mmapping the file");
        exit(EXIT_FAILURE);
    }

    int min = map[0], max = map[0];
    unsigned long histo[33];
    printf("FL\n");
    for (i = 0; i < 33; ++i)  histo[i]=0; 
    for (i = 0; i < fsize / 4; ++i)  {
        histo[BITLEN(map[i])]++; 
        if(min > map[i]) min = map[i];
        if(max < map[i]) max = map[i];
    }
    for (i = 0; i < 33; ++i)  printf(" %lu:%lu, ",i, histo[i]); 
    printf("\nmin %d max %d\n", min, max);

    printf("\nFOR-FL\n");
    for (i = 0; i < 33; ++i)  histo[i]=0; 
    for (i = 0; i < fsize / 4; ++i)  {
        histo[BITLEN(map[i]-min)]++; 
    }
    for (i = 0; i < 33; ++i)  printf(" %lu:%lu, ",i, histo[i]); 

    printf("\nDELTA_FL\n");
    min = map[0] - map[1]; max = map[0]-map[1];
    for (i = 0; i < 33; ++i)  histo[i]=0; 
    for (i = 1; i < fsize / 4; ++i) {
        histo[BITLEN(map[i-1] - map[i])]++; 
        if(min > map[i-1] - map[i]) min = map[i-1] - map[i];
        if(max < map[i-1] - map[i]) max = map[i-1] - map[i];
    } 
    for (i = 0; i < 33; ++i)  printf(" %lu:%lu, ",i, histo[i]); 

    printf("\nDELTA_FOR_FL\n");
    for (i = 0; i < 33; ++i)  histo[i]=0; 
    for (i = 1; i < fsize / 4; ++i)  histo[BITLEN(map[i-1] - map[i] - min )]++; 
    for (i = 0; i < 33; ++i)  printf(" %lu:%lu, ",i, histo[i]); 

    printf("\nAAFL\n");
    unsigned long nblocks = fsize/(4*1024) + 1;
    int j;
    unsigned long cblock_p;
    unsigned long cblock = 0;

    for (i = 0; i < nblocks; ++i)  {
        cblock_p = 0;
        for (j = 0; j < 1024; ++j) 
            cblock_p = max(BITLEN(map[i*1024 +j]), cblock_p);
        cblock += cblock_p * sizeof(int) * 8; 
    }
    printf("Comp r %f \n", (double)fsize / (cblock + (nblocks) *(sizeof(int) +sizeof(char))));

    printf("\nDELTA_AAFL\n");

    cblock = 0;
    for (i = 0; i < nblocks; ++i)  {
        cblock_p = 0;
        for (j = 1; j < 1024; ++j) 
            cblock_p = max(BITLEN(map[i*1024 + j -1 ] - map[i*1024 + j]), cblock_p);
        cblock += cblock_p * sizeof(int) * 8; 
    }
    printf("Comp r %f \n", (double)fsize / (cblock + nblocks * ( sizeof(int) + sizeof(int) + sizeof(char))));

    if (munmap(map, fsize) == -1) {
        perror("Error un-mmapping the file");
    }
    close(fd);
    return 0;

}

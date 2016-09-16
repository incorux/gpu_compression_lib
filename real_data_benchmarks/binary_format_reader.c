#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

int main(int argc, char *argv[])
{
    int i;
    int fd;
    unsigned int *map;  /* mmapped array of int's */
    if (argc !=2) exit(0);

    struct stat st;
    stat(argv[1], &st);
    unsigned long fsize = st.st_size;
    printf("File size %d Number of ints %d\n", fsize, fsize / 4);

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

    /* Read the file int-by-int from the mmap
    */
    for (i = 0; i <= fsize / 4; ++i) {
        printf("%d\n", map[i]);
    }

    if (munmap(map, FILESIZE) == -1) {
        perror("Error un-mmapping the file");
    }
    close(fd);
    return 0;

}

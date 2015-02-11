#ifndef _TOOLS
#define _TOOLS true

#include <stdio.h>
#include <list>


// Errors and debug
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
void _cudaErrorCheck(const char *file, int line);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define cudaErrorCheck()  { _cudaErrorCheck(__FILE__, __LINE__); }

// Time measuring tools
typedef struct timeit_info
{
    float __elapsedTime; 
    cudaEvent_t __start;
    cudaEvent_t __stop;
    char *name;
} timit_info;

typedef std::list<timeit_info *> tiManager;

void tiStart                ( tiManager &manager);
void tiEnd                  ( tiManager &manager, const char * name);
void tiPreatyPrint          ( tiManager &manager);
void tiClear                ( tiManager &manager);
void tiPreatyPrintThrougput ( tiManager &manager, int data_size);

#define TIMEIT_SETUP() tiManager __tim__;
#define TIMEIT_START() tiStart(__tim__);
#define TIMEIT_END(name) tiEnd(__tim__, name);
#define TIMEIT_PRINT() tiPreatyPrint(__tim__); tiClear(__tim__);
#define TIMEIT_PRINT_THROUGPUT(data_size) tiPreatyPrintThrougput(__tim__, data_size); tiClear(__tim__);
#endif

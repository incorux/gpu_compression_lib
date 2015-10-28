inline int if_debug()
{
   char* GPU_DEBUG;
   GPU_DEBUG = getenv ("GPU_DEBUG");

   if (GPU_DEBUG != NULL)
       if (atoi(GPU_DEBUG) > 0) return 1;
   return 0;
}

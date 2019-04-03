#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "contain.h"
#include "normal.h"

inline float randBetween(float From, float To)
{
   return From + (float(rand())/RAND_MAX) * (To-From) ;
}


main(int argc, char *argv[])
{
  if (argc != 6)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutFileName NoDiscrete NoDiscreteValues NoContinuous NumberOfExamples ",argv[0]) ;
     return 1 ;
  }
  FILE *fout ;
  if ((fout = fopen(argv[1],"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }

  int NoDiscrete = atoi(argv[2]) ;
  if (NoDiscrete < 0)
    goto wrongCall ;
  int NoDiscreteValues = atoi(argv[3]) ;
  if (NoDiscreteValues < 0)
    goto wrongCall ;
  int NoContinuous = atoi(argv[4]) ;
  if (NoContinuous < 0)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[5]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
  

  srand((unsigned)time(NULL)) ;
  RNGinit() ;
  
  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j ;
  for (int i=0; i < NumberOfExamples ; i++)
  {


     fprintf(fout,"%10.2f  ",randBetween(0.0, 1.0)) ;

     for (j=0 ; j < NoDiscrete ; j++)
	   fprintf(fout,"%4d ", int(fabs(Normal(j, NoDiscreteValues))) % NoDiscreteValues +1) ;

     for (j=0 ; j < NoContinuous ; j++)
	   fprintf(fout,"%f  ", Normal(j,1.0)) ;

     fprintf(fout,"\n") ;

  }
  fclose(fout) ;

  return 0 ;

}


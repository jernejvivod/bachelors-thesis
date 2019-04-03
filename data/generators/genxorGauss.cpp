#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "contain.h"
#include "normal.h"


inline double randBetween(double From, double To)
{
   return From + (double(rand())/RAND_MAX) * (To-From) ;
}


int main(int argc, char *argv[])
{
  if (argc != 6)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutFileName NoInfAttr NoRandAttr NumberOfExamples StdDevNoise",argv[0]) ;
     return 1 ;
  }
  FILE *fout ;
  if ((fout = fopen(argv[1],"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }

  int NoInfAttr = atoi(argv[2]) ;
  if (NoInfAttr <= 0)
    goto wrongCall ;
  int NoRandAttr = atoi(argv[3]) ;
  if (NoRandAttr < 0)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[4]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
  double StdDevNoise = atof(argv[5]) ;
  if (StdDevNoise < 0 )
    goto wrongCall ;
  
  
  srand((unsigned)time(NULL)) ;
  RNGinit() ;
 

  int NoDiscrete = NoInfAttr + NoRandAttr ;
  marray<int> D(NoDiscrete) ;
  double Function ;

  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j, sum ;
  for (int i=0; i < NumberOfExamples ; i++)
  {
     for (j=0; j < NoDiscrete ; j++)
       D[j] = rand() % 2 ;

     sum = 0 ;
     for (j=0 ; j < NoInfAttr ; j++)
       sum += D[j] ;

       if (sum % 2 == 0)
         Function = randBetween(0.0, 0.5) ;
       else
         Function = randBetween(0.5, 1.0) ;
     
     Function += Normal(0, StdDevNoise) ;
     
     fprintf(fout,"%10.4f  ",Function) ;

     for (j=0 ; j < NoDiscrete ; j++)
	fprintf(fout,"%4d ",int(D[j]+1)) ;


     fprintf(fout,"\n") ;

  }
  fclose(fout) ;

  return 0 ;

}
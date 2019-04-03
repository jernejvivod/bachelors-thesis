#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "contain.h"


inline float randBetween(float From, float To)
{
   return From + (float(rand())/RAND_MAX) * (To-From) ;
}


main(int argc, char *argv[])
{
  if (argc != 7)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutFileName Mod NoInfAttr NoRandAttr NumberOfExamples PercentNoise",argv[0]) ;
     return 1 ;
  }
  FILE *fout ;
  if ((fout = fopen(argv[1],"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }

  int Mod = atoi(argv[2]) ;
  if (Mod < 2)
    goto wrongCall ;
  int NoInfAttr = atoi(argv[3]) ;
  if (NoInfAttr <= 0)
    goto wrongCall ;
  int NoRandAttr = atoi(argv[4]) ;
  if (NoRandAttr < 0)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[5]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
  double PercentNoise = atof(argv[6]) ;
  if (PercentNoise < 0 || PercentNoise > 100)
    goto wrongCall ;
 

  srand((unsigned)time(NULL)) ;

  int NoContinuous = NoInfAttr + NoRandAttr ;
  int NoDiscrete =NoContinuous ;
  
  marray<double> C(NoContinuous) ;
  marray<int> D(NoDiscrete) ;
  double Function ;
  int sum ;
  
  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j ;
  for (int i=0; i < NumberOfExamples ; i++)
  {
     for (j=0 ; j < NoDiscrete; j++)
     {
       D[j] = rand() % Mod ;
       C[j] = D[j] ;
     }

     sum = 0 ;  
     for (j=0 ; j < NoInfAttr ; j++)
        sum+=D[j] ;
  
     if (rand() % 100 < PercentNoise)
       Function = double(rand() % Mod) ;
     else
       Function = double(sum % Mod) ;


     fprintf(fout,"%10.2f  ",Function) ;

     for (j=0 ; j < NoDiscrete ; j++)
	   fprintf(fout,"%4d ",int(D[j]+1)) ;

     for (j=0 ; j < NoContinuous ; j++)
	   fprintf(fout,"%10.2f ", C[j]) ;

     fprintf(fout,"\n") ;

  }
  fclose(fout) ;

  return 0 ;

}


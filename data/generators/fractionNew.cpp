#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "contain.h"
#include "utils.h"



int main(int argc, char *argv[])
{
  if (argc != 6)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s DomainName NoInfAttr NoRandAttr NumberOfExamples ProportionNoise",argv[0]) ;
     exit(1) ;
  }

  FILE *fout ;
  char buf[MaxPath] ;
  sprintf(buf,"%s.data",argv[1]) ;
  if ((fout = fopen(buf,"w")) == NULL)
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
  double ProportionNoise = atof(argv[5]) ;
  if (ProportionNoise < 0 || ProportionNoise > 1)
    goto wrongCall ;

  randSeed((unsigned)time(NULL)) ;

  int NoContinuous = NoInfAttr + NoRandAttr ;
  marray<double> C(NoContinuous) ;
  double sum, Function, ifunct ;

  int j ;
  for (int i=0; i < NumberOfExamples ; i++)
  {
     for (j=0 ; j < NoContinuous; j++)
       C[j] = randBetween(0.0, 1.0)  ;

     sum=0.0 ;
     for (j=0 ; j < NoInfAttr ; j++)
       sum += C[j] ;

    if (randBetween(0.0,1.0) < ProportionNoise)
      Function = randBetween(0.0, 1.0) ;
    else   
      Function = modf (sum, &ifunct) ;

     for (j=0 ; j < NoContinuous ; j++)
       fprintf(fout,"%g,", C[j]) ;

	 fprintf(fout,"%g\n",Function) ;

  }
  fclose(fout) ;

  return 0 ;

}
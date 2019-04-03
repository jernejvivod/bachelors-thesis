#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "contain.h"
#include "normal.h"

inline double randBetween(double From, double To)
{
   return From + (double(rand())/RAND_MAX) * (To-From) ;
}


main(int argc, char *argv[])
{
  if (argc != 6)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutFileName NoInfAttr NoRandAttr NumberOfExamples StdDevNoise",argv[0]) ;
     exit(1) ;
  }
  FILE *fout ;
  if ((fout = fopen(argv[1],"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     exit(1) ;
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
  double StdDev = atof(argv[5]) ;
  if (StdDev < 0.0)
    goto wrongCall ;
  

  srand((unsigned)time(NULL)) ;
  RNGinit() ;
  
  int NoContinuous = NoInfAttr + NoRandAttr ;
  marray<double> C(NoContinuous) ;
  double sum, Function, ifunct ;

  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j ;
  for (int i=0; i < NumberOfExamples ; i++)
  {
     for (j=0 ; j < NoContinuous; j++)
       C[j] = randBetween(0.0, 1.0)  ;

     sum=0.0 ;
     for (j=0 ; j < NoInfAttr ; j++)
       sum += C[j] ;

     Function = modf (sum, &ifunct) + Normal(0, StdDev) ;

     fprintf(fout,"%10.4f  ",Function) ;


     for (j=0 ; j < NoContinuous ; j++)
       fprintf(fout,"%10.4f ", C[j]) ;

     fprintf(fout,"\n") ;

  }
  fclose(fout) ;

  return 0 ;

}
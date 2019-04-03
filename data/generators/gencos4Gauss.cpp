#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "contain.h"
#include "normal.h"

const double PI = 3.1415926535 ;

inline double randBetween(double From, double To)
{
   return From + (double(rand())/RAND_MAX) * (To-From) ;
}


main(int argc, char *argv[])
{
  if (argc != 5)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutFileName NoRandAttr NumberOfExamples StdDevNoise",argv[0]) ;
     return 1 ;
  }
  FILE *fout ;
  if ((fout = fopen(argv[1],"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }

  int NoRandAttr = atoi(argv[2]) ;
  if (NoRandAttr < 0)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[3]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
  double StdDevNoise = atof(argv[4]) ;
  if (StdDevNoise < 0)
    goto wrongCall ;
  


  srand((unsigned)time(NULL)) ;
  RNGinit() ;


  int NoContinuous = 3 + NoRandAttr ;
  marray<double> C(NoContinuous) ;
  double Function ;

  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j ;
  for (int i=0; i < NumberOfExamples ; i++)
  {
     for (j=0 ; j < NoContinuous; j++)
       C[j] = randBetween(0.0, 1.0)  ;

      Function = (-2.0 * C[1] + 3.0 * C[2]) * cos(C[0]*4.0*PI) + Normal(0, StdDevNoise) ;

     fprintf(fout,"%10.4f  ",Function) ;

     for (j=0 ; j < NoContinuous ; j++)
        fprintf(fout,"%10.4f ", C[j]) ;

     fprintf(fout,"\n") ;
                                                         }
  fclose(fout) ;

  return 0 ;

}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "contain.h"

const double PI = 3.1415926535 ;

#define sqr(a) (a*a)

inline double randBetween(double From, double To)
{
   return From + (double(rand())/RAND_MAX) * (To-From) ;
}


main(int argc, char *argv[])
{
  if (argc != 5)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutFileName NoRandAttr NumberOfExamples PercentNoise",argv[0]) ;
     return 1 ;
  }

  FILE *fout ;
  char buf[MaxPath] ;
  sprintf(buf,"%s.dat",argv[1]) ;
  if ((fout = fopen(buf,"w")) == NULL)
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
  double PercentNoise = atof(argv[4]) ;
  if (PercentNoise < 0 || PercentNoise > 100)
    goto wrongCall ;
  


  srand((unsigned)time(NULL)) ;


  int NoContinuous = 2 + NoRandAttr ;
  marray<double> C(NoContinuous) ;
  double Function ;

  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j ;
  for (int i=0; i < NumberOfExamples ; i++)
  {
     for (j=0 ; j < NoContinuous; j++)
       C[j] = randBetween(-1.0, 1.0)  ;

     if (rand() % 100 < PercentNoise)
       Function = cos(2*PI*(sqr(randBetween(-1.0, 1.0))+3.0*sqr(randBetween(-1.0,1.0)))) ;
     else
       Function = cos(2*PI*(sqr(C[0])+3.0*sqr(C[1]))) ;

     fprintf(fout,"%10.5f  ",Function) ;

     for (j=0 ; j < NoContinuous ; j++)
        fprintf(fout,"%10.5f ", C[j]) ;

     fprintf(fout,"\n") ;
                                                         }
  fclose(fout) ;

   // generate description
  sprintf(buf,"%s.dsc",argv[1]) ;
  if ((fout = fopen(buf,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }

  fprintf(fout, "%d\nCosinus nice with %d random\n", 1+NoContinuous,NoRandAttr) ;
  fprintf(fout,"0 0.0 0.0\n") ;

  for (j=0 ; j < 2 ; j++)
  {

     fprintf(fout,"A%d\n",j+1) ;
     fprintf(fout,"0 0.0 0.0\n") ;
  }
  for (j=0 ; j < NoRandAttr ; j++)
  {
     fprintf(fout,"R%d\n",j+1) ;
     fprintf(fout,"0 0.0 0.0\n") ;
  }

  fclose(fout) ;

  // generate split file: all examples are for training
  sprintf(buf,"%s.99s",argv[1]) ;
  if ((fout = fopen(buf,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }

  for (i=0; i < NumberOfExamples ; i++)
    fprintf(fout,"0\n") ;
   
 fclose(fout) ;

  return 0 ;

}
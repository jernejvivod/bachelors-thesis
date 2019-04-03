#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "contain.h"


inline double randBetween(double From, double To)
{
   return From + (double(rand())/RAND_MAX) * (To-From) ;
}


main(int argc, char *argv[])
{
  if (argc != 6)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s DomainName NoInfAttr NoRandAttr NumberOfExamples PercentNoise",argv[0]) ;
     exit(1) ;
  }

  FILE *fout ;
  char buf[MaxPath] ;
  sprintf(buf,"%s.dat",argv[1]) ;
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
  double PercentNoise = atof(argv[5]) ;
  if (PercentNoise < 0 || PercentNoise > 100)
    goto wrongCall ;
  

  srand((unsigned)time(NULL)) ;


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

    if (rand() % 100 < PercentNoise)
      Function = randBetween(0.0, 1.0) ;
    else   
      Function = modf (sum, &ifunct) ;

     fprintf(fout,"%10.4f  ",Function) ;


     for (j=0 ; j < NoContinuous ; j++)
       fprintf(fout,"%10.4f ", C[j]) ;

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

  fprintf(fout, "%d\nFraction-%d with %d random\n", 1+NoContinuous,NoInfAttr,NoRandAttr) ;
  fprintf(fout,"0 0.0 0.0\n") ;

  for (j=0 ; j < NoInfAttr ; j++)
  {
     fprintf(fout,"I%d\n",j+1) ;
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
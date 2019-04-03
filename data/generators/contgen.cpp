#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "contain.h"


inline float randBetween(float From, float To)
{
   return From + (float(rand())/RAND_MAX) * (To-From) ;
}


main(int argc, char *argv[])
{
  if (argc != 3)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutFileName NumberOfExamples",argv[0]) ;
     exit(1) ;
  }
  FILE *fout ;
  if ((fout = fopen(argv[1],"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     exit(1) ;
  }

  int NumberOfExamples = atoi(argv[2]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;

  srand((unsigned)time(NULL)) ;


  int NoDiscrete = 4 ;
  int NoContinuous = 2 ;
  marray<int> D(NoDiscrete) ;
  marray<float> C(NoContinuous) ;
  float Function ;

  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j ;
  for (int i=0; i < NumberOfExamples ; i++)
  {
     D[0] = rand() % 2 ;
     D[1] = rand() % 2 ;
     D[2] = rand() % 2 ;
     D[3] = rand() % 2 ;
     C[0] = randBetween(0.0, 1.0)  ;
     C[1] = randBetween(0.0, 1.0)  ;

     if ((D[0]+D[1]) % 2 == 0)
       Function = randBetween(0.0, 0.5) ;
     else
       Function = randBetween(0.5, 1.0) ;

     fprintf(fout,"%10.3f  ",Function) ;

     for (j=0 ; j < NoDiscrete ; j++)
	fprintf(fout,"%4d ",int(D[j]+1)) ;

     for (j=0 ; j < NoContinuous ; j++)
	fprintf(fout,"%10.3f ", C[j]) ;

     fprintf(fout,"\n") ;

  }
  fclose(fout) ;

  return 0 ;

}
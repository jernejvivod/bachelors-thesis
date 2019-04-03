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


  int NoDiscrete = 2 ;
  int NoContinuous = 5 ;
  marray<int> D(NoDiscrete) ;
  marray<float> C(NoContinuous) ;
  double Function, wholePart ;

  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j ;
  for (int i=0; i < NumberOfExamples ; i++)
  {
//     for (j=0 ; j < NoDiscrete; j++)
//       D[j] = rand() % 3 ;
     for (j=0 ; j < NoContinuous; j++)
       C[j] = randBetween(0.0, 1.0)  ;


     Function = modf(C[0] + C[1], &wholePart) ;

     D[0] = rand % 2;
     if (Function < 0.5)
       if (D[0] == 0)  D[1] = 0 ;
       else D[1] = 1 ;
     else
       if (D[0] == 0)  D[1] = 1 ;
       else D[1] = 0 ;


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
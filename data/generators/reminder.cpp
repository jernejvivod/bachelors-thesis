#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "contain.h"


inline float randBetween(float From, float To)
{
   return From + (float(rand())/RAND_MAX) * (To-From) ;
}


main(int argc, char *argv[])
{
  if (argc != 5)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutFileName NumberOfExamples NumberOfInformative NumberOfRandom ",argv[0]) ;
     exit(1) ;
  }

  FILE *fout, *dscOut ;
  char DataName[256], DscName[256] ; ;
  strcpy(DataName, argv[1]) ;
  strcat(DataName,".dat") ;

  if ((fout = fopen(DataName,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",DataName) ;
     exit(1) ;
  }

  int NumberOfExamples = atoi(argv[2]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;

  int NumberOfInformative = atoi(argv[3]) ;
  if (NumberOfInformative <= 0)
    goto wrongCall ;

  int NumberOfRandom = atoi(argv[4]) ;
  if (NumberOfRandom < 0)
    goto wrongCall ;


  srand((unsigned)time(NULL)) ;


  int NoContinuous = NumberOfInformative + NumberOfRandom ;

  marray<float> C(NoContinuous) ;
  double Function, ifunct, dTemp ;

  fprintf(fout, "%d\n",NumberOfExamples) ;
  int i,j ;
  for (i=0; i < NumberOfExamples ; i++)
  {
     for (j=0 ; j < NoContinuous; j++)
       C[j] = randBetween(0.0, 1.0)  ;

     dTemp = C[0] ;
     for (j=1 ; j < NumberOfInformative ; j++)
        dTemp += C[j] ;
     Function = modf (dTemp, &ifunct) ;

     fprintf(fout,"%10.6f  ",Function) ;

     for (j=0 ; j < NoContinuous ; j++)
       fprintf(fout,"%10.6f ", C[j]) ;

     fprintf(fout,"\n") ;

  }
  fclose(fout) ;

  strcpy(DscName, argv[1]) ;
  strcat(DscName,".dsc") ;

  if ((dscOut = fopen(DscName,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",DscName) ;
     exit(1) ;
  }

  fprintf(dscOut, "%d \nF \n0 0.0 0.0 \n",NoContinuous+1) ;
  for (i=1 ; i <= NumberOfInformative ; i++)
     fprintf(dscOut,"I%d \n0 0.0 0.0 \n",i) ;
  for (i=1 ; i <= NumberOfRandom ; i++)
     fprintf(dscOut,"R%d \n0 0.0 0.0 \n",i) ;

  fclose(dscOut) ;

  return 0 ;

}

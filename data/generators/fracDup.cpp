#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "contain.h"


inline double randBetween(double From, double To)
{
   return From + (double(rand())/RAND_MAX) * (To-From) ;
}


int main(int argc, char *argv[])
{
  if (argc != 9)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s DomainName NoInfAttr NoDuplicatedInf NoCopyInf NoRandAttr NoDuplicatedRand NoCopyRand NumberOfExamples",argv[0]) ;
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

  int NoInfAttr = atoi(argv[2]) ;
  if (NoInfAttr <= 0)
    goto wrongCall ;
  int NoDuplInf = atoi(argv[3]) ;
  if (NoInfAttr < NoDuplInf)
    goto wrongCall ;
  int NoCopyInf = atoi(argv[4]) ;
  int NoRndAttr = atoi(argv[5]) ;
  if (NoRndAttr < 0)
    goto wrongCall ;
  int NoDuplRnd = atoi(argv[6]) ;
  if (NoRndAttr < NoDuplRnd)
    goto wrongCall ;
  int NoCopyRnd = atoi(argv[7]) ;
  
  int NumberOfExamples = atoi(argv[8]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
    
  srand((unsigned)time(NULL)) ;


  marray<double> C(NoInfAttr), R(NoRndAttr) ;

  double CFunction, ifunct ;

  // generate data
  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j, iC ;
  double sum ;
  for (int i=0; i < NumberOfExamples ; i++)
  {
     for (j=0; j < NoInfAttr ; j++)
       C[j] = randBetween(0,1) ;

     for (j=0; j < NoRndAttr ; j++)
       R[j] = randBetween(0,1) ;

     sum = 0 ;
     for (j=0 ; j < NoInfAttr ; j++)
       sum += C[j] ;

     CFunction = modf (sum, &ifunct) ;
     
     fprintf(fout,"%10.4f  ",CFunction) ;

     for (j=0 ; j < NoInfAttr ; j++)
       fprintf(fout,"%10.4f ",C[j]) ;

     for (j=0 ; j < NoDuplInf ; j++)
        for (iC=0 ; iC < NoCopyInf; iC++)
             fprintf(fout,"%10.4f ",C[j]) ;


     for (j=0 ; j < NoRndAttr ; j++)
       fprintf(fout,"%10.4f ", R[j]) ;

     for (j=0 ; j < NoDuplRnd ; j++)
        for (iC=0 ; iC < NoCopyRnd; iC++)
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

  fprintf(fout, "%d\nFractionDuplicated-%d-%d-%d-%d-%d-%d\n0 0.0 0.0\n",
                 1+NoInfAttr+NoDuplInf*NoCopyInf+NoRndAttr+NoDuplRnd*NoCopyRnd,
                 NoInfAttr,NoDuplInf,NoCopyInf,NoRndAttr,NoDuplRnd,NoCopyRnd) ;

   for (j=0 ; j < NoInfAttr ; j++)
       fprintf(fout,"I%d\n0 0.0 0.0\n",j+1) ;

     for (j=0 ; j < NoDuplInf ; j++)
        for (iC=0 ; iC < NoCopyInf; iC++)
             fprintf(fout,"I%dC%d\n0 0.0 0.0\n",j+1,iC+1) ;

     for (j=0 ; j < NoRndAttr ; j++)
       fprintf(fout,"R%d\n0 0.0 0.0\n",j+1) ;

     for (j=0 ; j < NoDuplRnd ; j++)
        for (iC=0 ; iC < NoCopyRnd; iC++)
             fprintf(fout,"R%dC%d\n0 0.0 0.0\n",j+1,iC+1) ;

 fclose(fout) ;

   // generate split file: all examples are for training
  sprintf(buf,"%s.99s",argv[1]) ;
  if ((fout = fopen(buf,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s", buf) ;
     return 1 ;
  }

  for (i=0; i < NumberOfExamples ; i++)
    fprintf(fout,"0\n") ;
   
 fclose(fout) ;


  return 0 ;

}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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


  marray<int> D(NoInfAttr), R(NoRndAttr) ;

  int DFunction ;

  // generate data
  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j, sum, iC ;
  for (int i=0; i < NumberOfExamples ; i++)
  {
     for (j=0; j < NoInfAttr ; j++)
       D[j] = rand() % 2 ;

     for (j=0; j < NoRndAttr ; j++)
       R[j] = rand() % 2 ;

     sum = 0 ;
     for (j=0 ; j < NoInfAttr ; j++)
       sum += D[j] ;

     if (sum % 2 == 0)
         DFunction = 1 ;
       else
         DFunction = 2 ;
     
     fprintf(fout,"%5d  ",DFunction) ;

     for (j=0 ; j < NoInfAttr ; j++)
       fprintf(fout,"%4d ",int(D[j]+1)) ;

     for (j=0 ; j < NoDuplInf ; j++)
        for (iC=0 ; iC < NoCopyInf; iC++)
             fprintf(fout,"%4d ",int(D[j]+1)) ;


     for (j=0 ; j < NoRndAttr ; j++)
       fprintf(fout,"%4d ",int(R[j]+1)) ;

     for (j=0 ; j < NoDuplRnd ; j++)
        for (iC=0 ; iC < NoCopyRnd; iC++)
             fprintf(fout,"%4d ",int(R[j]+1)) ;

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

  fprintf(fout, "%d\nXORduplicated-%d-%d-%d-%d-%d-%d\n2\nfalse\ntrue\n",
                 1+NoInfAttr+NoDuplInf*NoCopyInf+NoRndAttr+NoDuplRnd*NoCopyRnd,
                 NoInfAttr,NoDuplInf,NoCopyInf,NoRndAttr,NoDuplRnd,NoCopyRnd) ;

   for (j=0 ; j < NoInfAttr ; j++)
       fprintf(fout,"I%d\n2\nfalse\ntrue\n",j+1) ;

     for (j=0 ; j < NoDuplInf ; j++)
        for (iC=0 ; iC < NoCopyInf; iC++)
             fprintf(fout,"I%dC%d\n2\nfalse\ntrue\n",j+1,iC+1) ;

     for (j=0 ; j < NoRndAttr ; j++)
       fprintf(fout,"R%d\n2\nfalse\ntrue\n",j+1) ;

     for (j=0 ; j < NoDuplRnd ; j++)
        for (iC=0 ; iC < NoCopyRnd; iC++)
             fprintf(fout,"R%dC%d\n2\nfalse\ntrue\n",j+1,iC+1) ;

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
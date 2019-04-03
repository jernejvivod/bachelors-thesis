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
  if (argc != 5)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutDomainName NoInfAttr NoAllAttr NumberOfExamples",argv[0]) ;
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
  if (NoInfAttr < 0)
    goto wrongCall ;
  int NoDiscrete = atoi(argv[3]) ; 
  int NoRndAttr = NoDiscrete - NoInfAttr;
  if (NoRndAttr < 0)
    goto wrongCall ;
  
  int NumberOfExamples = atoi(argv[4]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
 
  marray<int> D(NoDiscrete) ;
 
 
  srand( (unsigned)time( NULL ) );
 

  int DFunction, iTemp, iPos, i ;
  
  /*
  marray<int> generated(NumberOfExamples) ;
  for (i=0 ; i < NumberOfExamples ; i++)
     generated[i] = i ;

  // shuffle generated marray
  for (i=0 ; i < NumberOfExamples ; i++)
  {
     iTemp = generated[i] ;
     iPos = rand() % NumberOfExamples ;
     generated[i] = generated[iPos] ;
     generated[iPos] = iTemp ;
  }
*/
  // generate data
  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j, sum ;
  for (i=0; i < NumberOfExamples ; i++)
  {
     // random generation of data
     for (j=0; j < NoDiscrete ; j++)
       D[j] = rand() % 2 ;
     
  /*   // generate all the data but in random order
     for (j=0; j < NoDiscrete ; j++)
     {
        D[j] = generated[i] % 2 ;
        generated[i] /= 2 ;
     }
*/
     sum = 0 ;
     for (j=0 ; j < NoInfAttr ; j++)
       sum += D[j] ;

     DFunction = sum % 2 ;
     
     fprintf(fout,"%5d  ",DFunction+1) ;

     for (j=0 ; j < NoDiscrete ; j++)
       fprintf(fout,"%4d ",int(D[j]+1)) ;

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

  fprintf(fout, "%d\nParity%dof%d\n2\nfalse\ntrue\n", 1+NoDiscrete,NoInfAttr,NoDiscrete) ;

  for (j=0 ; j < NoInfAttr ; j++)
       fprintf(fout,"I%d\n2\nfalse\ntrue\n",j+1) ;

  for (j=0 ; j < NoRndAttr ; j++)
       fprintf(fout,"R%d\n2\nfalse\ntrue\n",j+1) ;

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
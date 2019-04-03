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
  if (argc != 4)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutDomainName NoRndAttr NumberOfExamples\n",argv[0]) ;
     //fprintf(stderr, "       (ClassMode: 1-classification, 0- regression; AttrMode: 1-nominal, 0-numerical)\n") ;
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

  int ClassMode = 1;
  int AttrMode = 1 ;

  int NoInfAttr = 6 ;
  int NoRndAttr = atoi(argv[2]) ;
  if (NoRndAttr < 0)
    goto wrongCall ;
  int NoDiscrete = NoRndAttr + NoInfAttr;
  int NumberOfExamples = atoi(argv[3]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
 
  marray<int> D(NoDiscrete) ;
 
 
  srand( (unsigned)time( NULL ) );
 

  int DFunction,  i ;
  
  // generate data
  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j  ;
  for (i=0; i < NumberOfExamples ; i++)
  {
     // random generation of data
     for (j=0; j < NoDiscrete ; j++)
       D[j] = rand() % 2 ;
     
     if (D[0] == 0 && D[1] == 0)
        DFunction = (D[2] + D[3]) % 2 ;
     else if (D[0] == 0 && D[1] == 1)
          DFunction = D[4] && D[5] ;
         else if (D[0] == 1 && D[1] == 0)
            DFunction = (D[4] + D[5]) %2 ;
            else if (D[0] == 1 && D[1] == 1)
               DFunction = D[2] &&  D[3]  ;

        
     fprintf(fout,"%5d  ",DFunction+ClassMode) ;

     for (j=0 ; j < NoDiscrete ; j++)
       fprintf(fout,"%4d ", D[j]+AttrMode) ;

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

  fprintf(fout, "%d\nMultiplexer(I1,I2)(XOR(I3,I4), I3&I4&I5, I5|I6, maj(I3,I4,I5,I6))\n", 1+NoDiscrete) ;
  
  fprintf(fout,"2\n0\n1\n") ;

  for (j=0 ; j < NoInfAttr ; j++)
  {
     fprintf(fout,"I%d\n",j+1) ;
     fprintf(fout,"2\n0\n1\n") ;
  }

  for (j=0 ; j < NoRndAttr ; j++)
  {
     fprintf(fout,"R%d\n",j+1) ;
     fprintf(fout,"2\n0\n1\n") ;
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
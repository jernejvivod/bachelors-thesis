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
     fprintf(stderr, "Usage: %s OutDomainName ClassMode AttrMode Modulo NoInfAttr NoRandomAttr NumberOfExamples PercentNoise\n",argv[0]) ;
     fprintf(stderr, "       (ClassMode: 1-classification, 0- regression; AttrMode: 1-nominal, 0-numerical)\n") ;
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

  int ClassMode = atoi(argv[2]) ;
  if (ClassMode < 0 || ClassMode > 1)
      goto wrongCall ;
  int AttrMode = atoi(argv[3]) ;
  if (AttrMode < 0 || AttrMode > 1)
      goto wrongCall ;
  int Modulo = atoi(argv[4]) ;
  if (Modulo < 1)
    goto wrongCall ;
  int NoInfAttr = atoi(argv[5]) ;
  if (NoInfAttr < 0)
    goto wrongCall ;
  int NoRndAttr = atoi(argv[6]) ; 
  int NoDiscrete = NoInfAttr + NoRndAttr;
  if (NoRndAttr < 0)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[7]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
 
  double PercentNoise = atof(argv[8]) ;
  if (PercentNoise < 0 || PercentNoise > 100)
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
  int j,k, sum ;
  for (i=0; i < NumberOfExamples ; i++)
  {
     // random generation of data
     for (j=0; j < NoDiscrete ; j++)
       D[j] = rand() % Modulo ;
     
  /*   // generate all the data but in random order
     for (j=0; j < NoDiscrete ; j++)
     {
        D[j] = generated[i] % 2 ;
        generated[i] /= 2 ;
     }
*/
     if (rand() % 100 < PercentNoise)
       DFunction = rand() % Modulo ;
     else {
       sum = 0 ;
       for (j=0 ; j < NoInfAttr ; j++)
         sum += D[j] ;

        DFunction = sum % Modulo ;
     }

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

  fprintf(fout, "%d\nModulo-%d with %d of %d\n", 1+NoDiscrete,Modulo,NoInfAttr,NoDiscrete) ;
  
  if (ClassMode)
  {
     fprintf(fout,"%d\n",Modulo) ;
     for (k=0 ; k < Modulo ; k++)
        fprintf(fout, "%d\n", k);
  }
  else 
     fprintf(fout,"0 0.0 0.0\n") ;

  for (j=0 ; j < NoInfAttr ; j++)
  {
     fprintf(fout,"I%d\n",j+1) ;
     if (AttrMode)
     {
       fprintf(fout,"%d\n",Modulo) ;
       for (k=0 ; k < Modulo ; k++)
         fprintf(fout, "%d\n", k);
      }
      else 
         fprintf(fout,"0 0.0 0.0\n") ;
  }

  for (j=0 ; j < NoRndAttr ; j++)
  {
     fprintf(fout,"R%d\n",j+1) ;
     if (AttrMode)
     {
       fprintf(fout,"%d\n",Modulo) ;
       for (k=0 ; k < Modulo ; k++)
         fprintf(fout, "%d\n", k);
      }
      else 
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
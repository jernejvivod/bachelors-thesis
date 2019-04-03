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
  int NoMustArgs = 6 ;
  if (argc < NoMustArgs+2)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutDomainName Modulo NoInfAttr NoRandomAttr NumberOfExamples ClassProbabilities\n",argv[0]) ;
     return 1 ;
  }
  int ClassMode = 1, AttrMode = 1 ; // classification and discrete attributes
  FILE *fout ;
  char buf[MaxPath] ;
  sprintf(buf,"%s.dat",argv[1]) ;
  if ((fout = fopen(buf,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }
  int Modulo = atoi(argv[2]) ;
  int NoInfAttr = atoi(argv[3]) ;
  int NoRndAttr = atoi(argv[4]) ; 
  int NumberOfExamples = atoi(argv[5]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
 
  int cIdx ;
  marray<double> cP(Modulo,0.0), cPsum(Modulo+1,0.0) ;
  for (cIdx=0 ; cIdx < Modulo; cIdx++) {
     cP[cIdx] = atof(argv[NoMustArgs+cIdx]);
     cPsum[cIdx+1] = cPsum[cIdx] + cP[cIdx] ;
  }
  srand( (unsigned)time( NULL ) );

  int NoDiscrete = NoInfAttr + NoRndAttr ;
  marray<int> D(NoDiscrete) ;
  int DFunction, i ;
  double CFunction ;

  // generate data
  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j, sum ;
  marray<int> classProb(Modulo,0) ;
  for (i=0; i < NumberOfExamples ; i++)
  {
     CFunction = randBetween(0,1.0) ;
     for (DFunction=0 ; DFunction < Modulo; DFunction++) 
         if (CFunction <= cPsum[DFunction+1])
             break ;
     classProb[DFunction]++ ;
     // random generation of data
     for (j=1; j < NoDiscrete ; j++)
       D[j] = rand() % Modulo ;

     sum = 0 ;
     for (j=1 ; j < NoInfAttr ; j++)
       sum += D[j] ;
     sum = sum % Modulo ;
     D[0] =  (DFunction - sum + Modulo) % Modulo ;

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
     for (j=0 ; j < Modulo ; j++)
        fprintf(fout, "%d\n", j);
  }
  else 
     fprintf(fout,"0 0.0 0.0\n") ;

  for (j=0 ; j < NoInfAttr ; j++)
  {
     fprintf(fout,"I%d\n",j+1) ;
     if (AttrMode)
     {
       fprintf(fout,"%d\n",Modulo) ;
       for (i=0 ; i < Modulo ; i++)
         fprintf(fout, "%d\n", i);
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
       for (i=0 ; i < Modulo ; i++)
         fprintf(fout, "%d\n", i);
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

  // print out class distribution
  printf("Class value's probabilities\n") ;
  for (j=1 ; j <= Modulo ; j++)
      printf("%d=%d(%.4f)  ",j,classProb[j-1],classProb[j-1]/double(NumberOfExamples)) ;
  printf("\n");

   return 0 ;

}
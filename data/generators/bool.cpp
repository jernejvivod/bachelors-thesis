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
     fprintf(stderr, "Usage: %s OutDomainName NoRandAttr NumberOfExamples PercentNoise",argv[0]) ;
     return 1 ;
  }
 
  char fileName[MaxFileNameLen] ;
  FILE *fout ;
  
  // open data file
  sprintf(fileName, "%s.dat",argv[1]) ;
  if ((fout = fopen(fileName,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",fileName) ;
     return 1 ;
  }
  
  int NoRandAttr = atoi(argv[2]) ;
  if (NoRandAttr < 0)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[3]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
  double PercentNoise = atof(argv[4]) ; 
  if (PercentNoise < 0 || PercentNoise > 100)
    goto wrongCall ;
    
  srand((unsigned)time(NULL)) ;

  int NoInfAttr = 4;
  int NoDiscrete = NoInfAttr + NoRandAttr ;
  marray<int> D(NoDiscrete) ;
  int Function ;

  fprintf(fout, "%d\n",NumberOfExamples) ;
  int i, j, sum ;
  for (i=0; i < NumberOfExamples ; i++)
  {
     for (j=0; j < NoDiscrete ; j++)
       D[j] = rand() % 2 ;

     sum = (D[0]+D[1]) % 2  ;

     if (rand() % 100 < PercentNoise)
       Function = rand() %2  ;
     else
         Function = sum || (D[2] && D[3])  ;
     
     fprintf(fout,"%4d   ", int(Function)+1) ;

     for (j=0 ; j < NoDiscrete ; j++)
	fprintf(fout,"%4d ",int(D[j]+1)) ;


     fprintf(fout,"\n") ;

  }
  fclose(fout) ;


  // open description file
  sprintf(fileName, "%s.dsc",argv[1]) ;
  if ((fout = fopen(fileName,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",fileName) ;
     return 1 ;
  }

   fprintf(fout,"%d\nClass\n2\nfalse\ntrue\n",NoDiscrete+1) ;

   for (j=0 ; j < NoInfAttr ; j++)
    	fprintf(fout,"A%d\n2\nfalse\ntrue\n",j+1 ) ;
   
   for (j=0 ; j < NoRandAttr ; j++)
    	fprintf(fout,"R%d\n2\nfalse\ntrue\n",j+1 ) ;

  fclose(fout) ;

  // generate split file: all examples are for training
  sprintf(fileName,"%s.99s",argv[1]) ;
  if ((fout = fopen(fileName,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",fileName) ;
     return 1 ;
  }

  for (i=0; i < NumberOfExamples ; i++)
    fprintf(fout,"0\n") ;
   
  fclose(fout) ;


  return 0 ;

}

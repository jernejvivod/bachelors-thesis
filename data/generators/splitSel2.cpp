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
     fprintf(stderr, "Usage: %s OutName NoInfAttr NoRandAttr NumberOfExamples",argv[0]) ;
     return 1 ;
  }
  FILE *fout ;
  char fileName[MaxFileNameLen] ;
  
  // open data file
  sprintf(fileName, "%s.dat",argv[1]) ;
  if ((fout = fopen(fileName,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",fileName) ;
     return 1 ;
  }

  int NoInfAttr = atoi(argv[2]) ;
  if (NoInfAttr <= 0)
    goto wrongCall ;
  int NoRandAttr = atoi(argv[3]) ;
  if (NoRandAttr < 0)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[4]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
  
  
  srand((unsigned)time(NULL)) ;


  int NoDiscrete = NoInfAttr + NoRandAttr + 2 + 2 ;
  marray<int> D(NoDiscrete) ;
  int Function ;

  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j, sum ;
  for (int i=0; i < NumberOfExamples ; i++)
  {
     for (j=0; j < NoDiscrete ; j++)
       D[j] = rand() % 2 ;

     sum = 0 ;
     for (j=0 ; j < NoInfAttr ; j++)
       sum += D[j] ;
     if (sum % 2 == 0)
       Function = D[NoInfAttr] && D[NoInfAttr+1]  ;
     else
       Function = D[NoInfAttr+2] && D[NoInfAttr+3]  ;

     fprintf(fout,"%4d   ",Function+1) ;

     for (j=0 ; j < NoDiscrete ; j++)
     	fprintf(fout,"%4d ", D[j]+1) ;


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
    	fprintf(fout,"Xor%d\n2\nfalse\ntrue\n",j+1 ) ;
   for (j=0 ; j < 4 ; j++)
    	fprintf(fout,"A%d\n2\nfalse\ntrue\n",j+1 ) ;
   for (j=0 ; j < NoRandAttr ; j++)
    	fprintf(fout,"R%d\n2\nfalse\ntrue\n",j+1 ) ;

  return 0 ;

}
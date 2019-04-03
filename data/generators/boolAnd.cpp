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
  if (argc != 6)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s Type OutDomainName NoRandAttr NumberOfExamples PercentNoise\n",argv[0]) ;
	 fprintf(stderr, "       Type: 1=A1&A2|A1&A3, 2=A1&A2|A3&A4, 3=A1&A2|A3xorA4\n") ;
     return 1 ;
  }
 
  char fileName[MaxFileNameLen] ;
  FILE *fout ;
   
  int type = atoi(argv[1]) ;
  // open data file
  sprintf(fileName, "%s.dat",argv[2]) ;
  if ((fout = fopen(fileName,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",fileName) ;
     return 1 ;
  }
  
  int NoRandAttr = atoi(argv[3]) ;
  if (NoRandAttr < 0)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[4]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
  double PercentNoise = atof(argv[5]) ; 
  if (PercentNoise < 0 || PercentNoise > 100)
    goto wrongCall ;
    
  srand((unsigned)time(NULL)) ;

  int NoInfAttr ;
  if (type==1)  NoInfAttr = 3;
  else NoInfAttr = 4 ;

  int NoDiscrete = NoInfAttr + NoRandAttr ;
  marray<int> D(NoDiscrete) ;
  int Function ;

  fprintf(fout, "%d\n",NumberOfExamples) ;
  int i, j ;
  for (i=0; i < NumberOfExamples ; i++)
  {
     for (j=0; j < NoDiscrete ; j++)
       D[j] = rand() % 2 ;

     if (rand() % 100 < PercentNoise)
       Function = rand() %2  ;
     else 
		 switch (type) {
	     case 1: Function = (D[0] && D[1]) || (D[0] && D[2])  ;
			     break ;
		 case 2: Function = (D[0] && D[1]) || (D[2] && D[3])  ;
			     break ;
		 case 3: Function = (D[0] && D[1]) || ((D[2]+D[3])%2)  ;
			     break ;
		 }
     
     fprintf(fout,"%4d   ", int(Function)+1) ;

     for (j=0 ; j < NoDiscrete ; j++)
    	fprintf(fout,"%4d ",int(D[j]+1)) ;
     fprintf(fout,"\n") ;

  }
  fclose(fout) ;

  // open description file
  sprintf(fileName, "%s.dsc",argv[2]) ;
  if ((fout = fopen(fileName,"w")) == NULL)  {
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
  sprintf(fileName,"%s.99s",argv[2]) ;
  if ((fout = fopen(fileName,"w")) == NULL)  {
     fprintf(stderr,"Cannot write to file %s",fileName) ;
     return 1 ;
  }

  for (i=0; i < NumberOfExamples ; i++)
    fprintf(fout,"0\n") ;
   
  fclose(fout) ;


  return 0 ;

}

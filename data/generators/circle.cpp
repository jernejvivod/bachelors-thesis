#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include "contain.h"
#include "utils.h"
#include "circle.h"


booleanT inCircle(double x, double y, double x0, double y0, double r) {
	if ( sqr(x-x0)+sqr(y-y0) < sqr(r))
		return mTRUE ;
	else 
		return mFALSE ;
}

int main(int argc, char *argv[])
{
  if (argc != 5)
  {
   wrongCall:
     fprintf(stderr, "\nUsage: %s OutDomainName NumberOfRandomAttr NumberOfExamples NoiseLevel\n", argv[0]) ;
     return 1 ;
  }
 
  char fileName[MaxFileNameLen] ;
  FILE *fout ;
  
  // open data file
  sprintf(fileName, "%s.data",argv[1]) ;
  if ((fout = fopen(fileName,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",fileName) ;
     return 1 ;
  }
  
  int NoRandomAttr = atoi(argv[2]) ;
  if (NoRandomAttr < 0)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[3]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
  double noiseLevel = atof(argv[4]) ;
  if (noiseLevel < 0.0 || noiseLevel>1.0)
    goto wrongCall ;
    
  srand((unsigned)time(NULL)) ;

  int NoInfAttr = 2;
  //marray<double> prob(NoInfAttr+1) ;
  //prob[0] = 0.0 ; prob[1] = 0.9 ; prob[2] = 0.8 ; prob[3] = 0.7 ; 

  int noAttr = NoInfAttr+NoRandomAttr ;
  marray<double> A(noAttr+1);
  
  
  int i, j;
  for (i=0; i < NumberOfExamples ; i++)
  {
      A[1] =  randBetween(0.0,1.0)  ;
      A[2] =  randBetween(0.0,1.0)  ;

	  A[0] = (inCircle(A[1],A[2],0.5,0.5,sqrt(1.0/2.0/M_PI)) ?  1.0 : 0.0) ;
      if (randBetween(0.0, 1.0) < noiseLevel)
		  A[0] = fabs(1.0 - A[0]) ;

	  for (j = 1+NoInfAttr;  j <=noAttr; j++)
	      A[j] =  randBetween(0.0,1.0)  ;

      for (j=1 ; j <= noAttr ; j++)
   	    fprintf(fout,"%f, ",A[j]) ;
      fprintf(fout,"%d\n",int(A[0])) ;
  }
  fclose(fout) ;

  // names file
  sprintf(fileName, "%s.names",argv[1]) ;
  if ((fout = fopen(fileName,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",fileName) ;
     return 1 ;
  }

  //fprintf(fout,"Class.\n\nClass:  1,0.\n") ;
  fprintf(fout,"1,0.\n") ;
  for (j=1 ; j <= NoInfAttr ; j++)   
	  fprintf(fout,"I%d:  continuous.\n",j) ;
  for (j = NoInfAttr+1 ;  j <= noAttr; j++)
	  fprintf(fout,"R%d:  continuous.\n", j-NoInfAttr) ;
  fclose(fout) ;

  return 0 ;
}

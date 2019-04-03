#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include "contain.h"
#include "utils.h"
#include "modGroup.h"


booleanT inCircle(double x, double y, double x0, double y0, double r) {
	if ( sqr(x-x0)+sqr(y-y0) < sqr(r))
		return mTRUE ;
	else 
		return mFALSE ;
}

int main(int argc, char *argv[])
{
  if (argc != 8)
  {
   wrongCall:
     fprintf(stderr, "\nUsage: %s OutDomainName M N NumberOfRandomAttr NumberOfExamples NoiseLevel PortionOfClass1 \n", argv[0]) ;
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
  
  int M = atoi(argv[2]) ;
  if (M < 2)
    goto wrongCall ;
  int N = atoi(argv[3]) ;
  if (N < 2)
    goto wrongCall ;
  int NoRandomAttr = atoi(argv[4]) ;
  if (NoRandomAttr < 0)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[5]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
  double noiseLevel = atof(argv[6]) ;
  if (noiseLevel < 0.0 || noiseLevel>1.0)
    goto wrongCall ;
    
  double portionOfClass1 = atof(argv[7]) ;
  if (portionOfClass1 < 0.0 || portionOfClass1>1.0)
    goto wrongCall ;

  randSeed((unsigned)time(NULL)) ;
  
  int NoInfAttr = 2 ;
  int noAttr = NoInfAttr+NoRandomAttr ;
  marray<double> A(noAttr+1);
  
  
  int i, j, classValue;
  i=0;
  int sumClass1 = 0 ;
  while (i < NumberOfExamples) {
  
	  for (j=1 ; j <= noAttr ; j++) {
		 A[j] = randBetween(0.0, 1.0) ;
	  } 

	  classValue = (int(A[1] * M) + int(A[2] * N)) % 2 ;
      sumClass1 += classValue ;


	  if (classValue==0 || (classValue==1 && double(sumClass1)/NumberOfExamples <= portionOfClass1)) {
  	    if (randBetween(0.0, 1.0) < noiseLevel)
		    classValue = (1 + classValue) % 2 ;

		for (j=1 ; j <= noAttr ; j++)
   	      fprintf(fout,"%f, ",A[j]) ;
        fprintf(fout,"%d\n",classValue) ;
		
		++i ;
	  }
  }
  fclose(fout) ;

  // names file
  sprintf(fileName, "%s.names",argv[1]) ;
  if ((fout = fopen(fileName,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",fileName) ;
     return 1 ;
  }

  fprintf(fout,"0, 1.\n") ;
  for (j=1 ; j <= NoInfAttr ; j++)   
	  fprintf(fout,"I%d:  continuous.\n",j) ;
  for (j = NoInfAttr+1 ;  j <= noAttr; j++)
	  fprintf(fout,"R%d:  continuous.\n", j-NoInfAttr) ;
  fclose(fout) ;

  return 0 ;
}

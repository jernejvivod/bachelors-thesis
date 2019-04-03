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
  if (argc != 7)
  {
   wrongCall:
     fprintf(stderr, "\nUsage: %s OutDomainName Mod NoImportantAttr NumberOfRandomAttr NumberOfExamples NoiseLevel\n", argv[0]) ;
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
  
  int Mod = atoi(argv[2]) ;
  if (Mod < 2)
    goto wrongCall ;
  int NoInfAttr = atoi(argv[3]) ;
  if (NoInfAttr < 0)
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
    
  randSeed((unsigned)time(NULL)) ;

  int noAttr = NoInfAttr+NoRandomAttr ;
  marray<double> A(noAttr+1);
  marray<int> centerA(noAttr+1);
  
  
  int i, j, classValue;
  for (i=0; i < NumberOfExamples ; i++)
  {
	  for (j=1 ; j <= noAttr ; j++) {
		 centerA[j] = randBetween(0, Mod) ;
		 A[j] = Mmax(double(centerA[j])/Mod, Mmin(double(centerA[j]+1)/Mod, randNormal((2.0*centerA[j]+1.0)/2.0/Mod, 1.0/Mod/6.0))) ;
	  } 

	  classValue = 0 ;
	  for (j=1 ; j <= NoInfAttr ; j++) 
		 classValue += centerA[j] ;
      classValue = classValue % Mod ;
	  if (randBetween(0.0, 1.0) < noiseLevel)
		  classValue = (randBetween(1,Mod) + classValue) % Mod ;

      for (j=1 ; j <= noAttr ; j++)
   	    fprintf(fout,"%f, ",A[j]) ;
      fprintf(fout,"%d\n",classValue) ;
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
  fprintf(fout,"0") ;
  for (j=1 ; j < Mod; j++)
	  fprintf(fout,",%d",j) ;
  fprintf(fout,".\n") ;
  for (j=1 ; j <= NoInfAttr ; j++)   
	  fprintf(fout,"I%d:  continuous.\n",j) ;
  for (j = NoInfAttr+1 ;  j <= noAttr; j++)
	  fprintf(fout,"R%d:  continuous.\n", j-NoInfAttr) ;
  fclose(fout) ;

  return 0 ;
}

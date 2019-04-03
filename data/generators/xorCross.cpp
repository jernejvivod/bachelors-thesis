#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include "contain.h"
#include "utils.h"
#include "xorCross.h"

enum dataFormatType { C45=0, WEKA=1 } ;

int main(int argc, char *argv[])
{
  if (argc != 8)
  {
   wrongCall:
     fprintf(stderr, "\nUsage: %s dataFormat outDomainName numberOfRandomAttr numberOfExamples noiseLevel margin crossWidth\n", argv[0]) ;
     return 1 ;
  }
 
  char fileName[MaxFileNameLen] ;
  FILE *fout ;
  int i, j;
  
  int dataFormat = atoi(argv[1]) ;
  if (dataFormat <0 || dataFormat>1)
      goto wrongCall ;
  
  int NoRandomAttr = atoi(argv[3]) ;
  if (NoRandomAttr < 0)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[4]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
  double noiseLevel = atof(argv[5]) ;
  if (noiseLevel < 0.0 || noiseLevel>1.0)
    goto wrongCall ;
  double margin = atof(argv[6]) ;
  if (margin < 0.0 || margin>0.5)
    goto wrongCall ;
  double crossWidth = atof(argv[7]) ;
  if (crossWidth < 0.0 || crossWidth>0.5 || crossWidth+margin/2.0 > 0.5)
    goto wrongCall ;
  srand((unsigned)time(NULL)) ;

  int NoInfAttr = 2;
  //marray<double> prob(NoInfAttr+1) ;
  //prob[0] = 0.0 ; prob[1] = 0.9 ; prob[2] = 0.8 ; prob[3] = 0.7 ; 
   
  int noAttr = NoInfAttr+NoRandomAttr ;
  marray<double> A(noAttr+1);

  randSeed(time( NULL ) );


  if (dataFormat==0) { //C4.5
    // names file
    sprintf(fileName, "%s.names",argv[2]) ;
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
    // open data file
    sprintf(fileName, "%s.data",argv[2]) ;
    if ((fout = fopen(fileName,"w")) == NULL)
    {
       fprintf(stderr,"Cannot write to file %s",fileName) ;
       return 1 ;
    }
  }
  else { // weka format
    sprintf(fileName, "%s.arff",argv[2]) ;
    if ((fout = fopen(fileName,"w")) == NULL)
    {
       fprintf(stderr,"Cannot write to file %s",fileName) ;
       return 1 ;
    }
    fprintf(fout,"@RELATION xorCross\n\n") ;
    for (j=1 ; j <= NoInfAttr ; j++)   
	   fprintf(fout,"@ATTRIBUTE I%d  numeric\n",j) ;
    for (j = NoInfAttr+1 ;  j <= noAttr; j++)
	  fprintf(fout,"@ATTRIBUTE R%d  numeric\n", j-NoInfAttr) ;
	fprintf(fout,"@ATTRIBUTE class  {1,0}\n") ;
	fprintf(fout,"\n\n@DATA\n") ;
  }


  // double areaOf1 = crossWidth *  (1.0 - margin + crossWidth) ;
  // double upperArea = crossWidth *  (0.5 - margin/2.0 + crosswidth) ;
  double probUpper = (0.5 - margin/2.0 + crossWidth)  / (1.0 - margin + crossWidth) ;
  for (i=0; i < NumberOfExamples ; i++)
  {
	  //A[1] = randBetween(0.0,1.0) ;
	  //A[2] = randBetween(0.0,1.0) ;
	  // generate data in upper right corner
      if (randBetween(0.0, 1.0)<probUpper) {
		   //vertical part
		   A[1] = randBetween(0.5 + margin/2.0, 0.5 + margin/2.0 + crossWidth) ;
  	       A[2] = randBetween(0.5 + margin/2.0 + crossWidth, 1.0) ;
	  }
	  else {
	    // horizontal part
	    A[1] = randBetween(0.5 + margin/2.0, 1.0) ;
  	    A[2] = randBetween(0.5 + margin/2.0, 0.5 + margin/2.0 + crossWidth) ;
	  }
	  // now we use transformation over the right axis
	  double section  = randBetween(0.0, 1.0) ;
	  if (section<0.25) {
	  	 // over A[1] axist
		 A[0] = 0.0 ;
		 A[1] = 1.0-A[1] ;
	  }
	  else if (section<0.5) {
	  	 // over A[2] axis
		 A[0] = 0.0 ;
		 A[2] = 1.0-A[2] ;
	  }
	  else if (section<0.75) {
	  	 // over A[1] and A[2] axis
		 A[0] = 1.0 ;
		 A[1] = 1.0 - A[1] ;
		 A[2] = 1.0 - A[2] ;
	  }
	  else {
		  A[0] = 1.0 ;
	  }	 
      if (randBetween(0.0, 1.0) < noiseLevel)
		  A[0] = fabs(1.0 - A[0]) ;

	  for (j = 1+NoInfAttr;  j <=noAttr; j++)
	      A[j] =  randBetween(0.0,1.0)  ;

      for (j=1 ; j <= noAttr ; j++)
   	    fprintf(fout,"%f, ",A[j]) ;
      fprintf(fout,"%d\n",int(A[0])) ;
  }
  fclose(fout) ;


  return 0 ;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include "contain.h"
#include "condInd.h"

double randBetween(double From, double To)
{
   return From + (double(rand())/RAND_MAX) * (To-From) ;
}


int main(int argc, char *argv[])
{
  if (argc != 4)
  {
   wrongCall:
     fprintf(stderr, "\nUsage: %s OutDomainName NumberOfRandomAttr NumberOfExamples\n", argv[0]) ;
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
    
  srand((unsigned)time(NULL)) ;

  int NoInfAttr = 4;
  marray<double> prob(NoInfAttr+1) ;
  prob[0] = 0.0 ; prob[1] = 0.6 ; prob[2] = 0.7 ; prob[3] = 0.8 ; prob[4] = 0.9 ;
  
  int noDiscrete = NoInfAttr+NoRandomAttr ;
  marray<int> D(noDiscrete+1);

  int i, j;
  for (i=0; i < NumberOfExamples ; i++)
  {
	  D[0] =  randBetween(0.0,1.0) < 0.5 ? 0 : 1 ;
	  for (j = 1 ;  j <=NoInfAttr; j++)
		 if (randBetween(0.0,1.0) <= prob[j])
		     D[j] = D[0] ;
		 else 
			 D[j] = abs(1-D[0]) ;
	  for (j = NoInfAttr+1 ;  j <= noDiscrete; j++)
	      D[j] =  (randBetween(0.0,1.0) < 0.5 ? 0 : 1) ;
  
      for (j=1 ; j <= noDiscrete ; j++)
   	    fprintf(fout,"%d, ",D[j]) ;
      fprintf(fout,"%d\n",D[0]) ;
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
	  fprintf(fout,"I%d:  1,0.\n",int(prob[j]*100.0+0.5)) ;
  for (j = NoInfAttr+1 ;  j <= noDiscrete; j++)
	  fprintf(fout,"R%d:  1,0.\n", j-NoInfAttr) ;
  fclose(fout) ;

  return 0 ;
}

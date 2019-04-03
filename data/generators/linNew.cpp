#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "contain.h"
#include "utils.h"

int main(int argc, char *argv[])
{
  if (argc != 6)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutFileName  NoLinAttr NoRandAttr NumberOfExamples propNoise",argv[0]) ;
     return 1 ;
  }
  
  FILE *fout ;
  char buf[MaxPath] ;
  sprintf(buf,"%s.data",argv[1]) ;
  if ((fout = fopen(buf,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }

  int NoLinAttr = atoi(argv[2]);
  int NoRandAttr = atoi(argv[3]) ;
  if (NoRandAttr < 0)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[4]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
  double propNoise = atof(argv[5]) ;
  if (propNoise < 0 || propNoise > 1)
    goto wrongCall ;
  

  randSeed((unsigned)time(NULL)) ;


  int NoContinuous = NoLinAttr + NoRandAttr ;
  marray<double> C(NoContinuous) ;
  double Function  ;
  int i,j,idx ;
  
  //fprintf(fout, "%d\n",NumberOfExamples) ;
  for (i=0; i < NumberOfExamples ; i++)
  {
     for (j=0 ; j < NoContinuous; j++)
       C[j] = randBetween(0.0, 1.0)  ;

	 Function = 0.0 ;
     if (randBetween(0.0,1.0) > propNoise) 
        for (idx=0;idx < NoLinAttr ; idx++) 
		    Function += (idx+1)*C[idx] ;
	 else 
        for (idx=0;idx < NoLinAttr ; idx++) 
		    Function += (idx+1)*randBetween(0.0, 1.0) ;
     
	 for (j=0 ; j < NoContinuous ; j++)
       fprintf(fout,"%g, ", C[j]) ;
     fprintf(fout,"%g  \n",Function) ;
  }
  fclose(fout) ;

  // generate description
  sprintf(buf,"%s.names",argv[1]) ;
  if ((fout = fopen(buf,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }
  fprintf(fout,"continuous.\n") ;
  for (j=1 ; j <= NoLinAttr ; j++)   
	  fprintf(fout,"L%d:  continuous.\n",j) ;
  for (j = NoLinAttr+1 ;  j <= NoContinuous; j++)
	  fprintf(fout,"R%d:  continuous.\n", j-NoLinAttr) ;
  fclose(fout) ;

  return 0 ;

}
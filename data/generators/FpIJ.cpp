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
  if (argc != 4)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutDomainName I J",argv[0]) ;
     return 1 ;
  }
  FILE *fout ;
  if ((fout = fopen(argv[1],"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }

  int I = atoi(argv[2]) ;
  int J = atoi(argv[3]) ;

  int NoDiscrete = 12 ;
  marray<int> D(NoDiscrete) ;
  int Function;
  int NumberOfExamples = 16384; // 4096 ;

  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j, sum ;
  for (int i=0; i < NumberOfExamples ; i++)
  {
     sum = i ;
     for (j=0; j < NoDiscrete ; j++)
     {
        D[j] = sum % 2 ;
        sum /= 2 ;
     }
     
     sum = 0 ;
     for (j=I-1 ; j < J ; j++)
       sum += D[j] ;

     Function = sum % 2 ;
     
     fprintf(fout,"%7d  ",Function+1) ;

     for (j=0 ; j < NoDiscrete ; j++)
	    fprintf(fout,"%4d ",int(D[j]+1)) ;


     fprintf(fout,"\n") ;

  }
  fclose(fout) ;

  return 0 ;

}
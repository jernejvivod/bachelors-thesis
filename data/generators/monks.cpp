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
     fprintf(stderr, "\nUsage: %s OutDomainName ProblemIdx NumberOfExamples PercentNoise\n", argv[0]) ;
     fprintf(stderr, "        (ProblemIdx: 1-3)\n",argv[0]) ;
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
  
  int ProblemIdx = atoi(argv[2]) ;
  if (ProblemIdx < 0 || ProblemIdx > 3)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[3]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
  double PercentNoise = atof(argv[4]) ; 
  if (PercentNoise < 0 || PercentNoise > 100)
    goto wrongCall ;
    
  srand((unsigned)time(NULL)) ;

  int NoInfAttr = 6;
  int NoDiscrete = NoInfAttr ;
  marray<int> D(NoDiscrete), noValues(NoDiscrete) ;
  noValues[0] = noValues[1] = noValues [3] = 3 ;
  noValues[2] = noValues[5] = 2 ;
  noValues[4] = 4 ;
  int Function ;

  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j,k, temp ;
  for (int i=0; i < NumberOfExamples ; i++)
  {
     for (j = 0 ;  j < NoInfAttr; j++)
       D[j] = rand() % noValues[j] ;

   
     if (rand() % 100 < PercentNoise)
       Function = rand() %2  ;
     else
     switch (ProblemIdx) {  ;
     case 1: if (D[0] == D[1]  || D[4] == 0) 
                Function = 1;
             else Function = 0 ;
             break ;
     case 2: temp = 0 ;
             for (j=0 ; j < 6 ; j++)
                if (D[j] == 0)
                   temp ++ ;
             if (temp == 2)
                Function = 1;
             else
                Function  = 0 ;
             break ;
     case 3:if ( (D[4] == 2 && D[3] == 0) || (D[4] != 3 && D[1] != 2) )
               Function = 1 ;
            else 
               Function = 0 ;
     }

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
   {
    	fprintf(fout,"A%d\n%d\n",j+1, noValues[j] ) ;
      for (k=1 ; k <= noValues[j] ; k++)
         fprintf(fout,"%d\n",k) ;
   }
 
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
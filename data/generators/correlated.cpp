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
     fprintf(stderr, "Usage: %s OutDomainName NoAttrValues NoRndAttr NumberOfExamples\n",argv[0]) ;
     //fprintf(stderr, "       (ClassMode: 1-classification, 0- regression; AttrMode: 1-nominal, 0-numerical)\n") ;
     return 1 ;
  }
  FILE *fout ;
  char buf[MaxPath] ;
  sprintf(buf,"%s.dat",argv[1]) ;
  if ((fout = fopen(buf,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }

  int ClassMode = 1 ;
  int AttrMode = 1 ;
  int NoInfAttr = 10 ;
  int NoAttrValues = atoi(argv[2]) ;
  int NoRndAttr = atoi(argv[3]) ;
  if (NoRndAttr < 0)
    goto wrongCall ;
  int NoDiscrete = NoRndAttr + NoInfAttr ;
  
  int NumberOfExamples = atoi(argv[4]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
 
  marray<int> D(NoDiscrete) ;
 
 
  srand( (unsigned)time( NULL ) );
 

  int DFunction, i ;
  
  // generate data
  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j, k ;
  for (i=0; i < NumberOfExamples ; i++)
  {
     
     DFunction = rand() % NoAttrValues ;
  
     // generation of important attributes
     for (j=0; j < NoInfAttr ; j++)
       if (rand() % 100 < 100 - 5*(j+1))
           D[j] = DFunction ;
       else do {
          D[j] = rand() % NoAttrValues ;
       } while (D[j] == DFunction) ;

   // generation of random attributes
     for (j=NoInfAttr ;  j < NoDiscrete ; j++)
        D[j] = rand() % NoAttrValues ;

     
     fprintf(fout,"%5d  ",DFunction+ClassMode) ;

     for (j=0 ; j < NoDiscrete ; j++)
       fprintf(fout,"%4d ", D[j]+AttrMode) ;

     fprintf(fout,"\n") ;

  }
  fclose(fout) ;

  // generate description
  sprintf(buf,"%s.dsc",argv[1]) ;
  if ((fout = fopen(buf,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }

  fprintf(fout, "%d\nCorreleation with %d attribute values and %d random attributes\n", NoDiscrete+1, NoAttrValues, NoRndAttr) ;
  
  fprintf(fout,"%d\n",NoAttrValues) ;
  for (k=0 ; k < NoAttrValues ; k++)
     fprintf(fout, "V%d\n", k);

  for (j=0 ; j < NoInfAttr ; j++)
  {
     fprintf(fout,"I%d\n",j+1) ;
     fprintf(fout,"%d\n",NoAttrValues) ;
     for (k=0 ; k < NoAttrValues ; k++)
        fprintf(fout, "V%d\n", k);
  }

  for (j=0 ; j < NoRndAttr ; j++)
  {
     fprintf(fout,"R%d\n",j+1) ;
     fprintf(fout,"%d\n",NoAttrValues) ;
     for (k=0 ; k < NoAttrValues ; k++)
       fprintf(fout, "V%d\n", k);
  }

  fclose(fout) ;

  // generate split file: all examples are for training
  sprintf(buf,"%s.99s",argv[1]) ;
  if ((fout = fopen(buf,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }

  for (i=0; i < NumberOfExamples ; i++)
    fprintf(fout,"0\n") ;
   


 fclose(fout) ;

  return 0 ;

}
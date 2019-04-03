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
  int NoMustArgs = 5 ;
  if (argc < NoMustArgs+2)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutDomainName NumberOfExamples NumberOfClasses NoRandomAttr ClassProbabilities\n",argv[0]) ;
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
  int NumberOfExamples = atoi(argv[2]) ;
  int NumberOfClasses = atoi(argv[3]) ;
  int NoClasses = atoi(argv[3]) ;
  int NoRndAttr = atoi(argv[4]) ;
  int cIdx ;
  marray<double> cP(NoClasses,0.0), cPsum(NoClasses+1,0.0) ;
  for (cIdx=0 ; cIdx < NoClasses; cIdx++) {
     cP[cIdx] = atof(argv[NoMustArgs+cIdx]);
     cPsum[cIdx+1] = cPsum[cIdx] + cP[cIdx] ;
  }
 
  srand( (unsigned)time( NULL ) );
  int ClassMode = 1, AttrMode = 1 ; // classification and discrete attributes

  int NoDiscrete = 3*NoClasses + NoRndAttr ;
  marray<int> D(NoDiscrete) ;
  int DFunction, i ;
  double CFunction ;

  // generate data
  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j, sum ;
  marray<int> classProb(NoClasses,0) ;
  for (i=0; i < NumberOfExamples ; i++)
  {
     CFunction = randBetween(0,1.0) ;
     for (DFunction=0 ; DFunction < NoClasses; DFunction++) 
         if (CFunction <= cPsum[DFunction+1])
             break ;
     classProb[DFunction]++ ;

     // random generation of data
     for (j=0; j < NoClasses ; j++) {
       if (DFunction == j && randBetween(0.0, 1.0) < 0.8 )
           D[j] = DFunction ;
       else D[j] = rand() % NoClasses ;
       D[j+NoClasses] = rand() % NoClasses ;
       D[j+2*NoClasses] = (D[j] - D[j+NoClasses]+NoClasses) % NoClasses ;
     }

     for (j=3*NoClasses ; j < 3*NoClasses+NoRndAttr ; j++)
       D[j] = rand() % NoClasses ;

     // output values
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

  fprintf(fout, "%d\nArtCost-%d-%d-%d\n", 1+NoDiscrete,NoClasses,2,NoRndAttr) ;
  
  if (ClassMode)
  {
     fprintf(fout,"%d\n",NoClasses) ;
     for (j=0 ; j < NoClasses ; j++)
        fprintf(fout, "%d\n", j);
  }
  else 
     fprintf(fout,"0 0.0 0.0\n") ;

  for (j=0 ; j < NoClasses ; j++)
  {
     fprintf(fout,"V%d\n",j+1) ;
     if (AttrMode)
     {
       fprintf(fout,"%d\n",NoClasses) ;
       for (i=0 ; i < NoClasses ; i++)
         fprintf(fout, "%d\n", i);
      }
      else 
         fprintf(fout,"0 0.0 0.0\n") ;
  }

    for (j=0 ; j < NoClasses ; j++)
  {
     fprintf(fout,"XOR-V%d-1\n",j+1) ;
     if (AttrMode)
     {
       fprintf(fout,"%d\n",NoClasses) ;
       for (i=0 ; i < NoClasses ; i++)
         fprintf(fout, "%d\n", i);
      }
      else 
         fprintf(fout,"0 0.0 0.0\n") ;
  }

    for (j=0 ; j < NoClasses ; j++)
  {
     fprintf(fout,"XOR-V%d-2\n",j+1) ;
     if (AttrMode)
     {
       fprintf(fout,"%d\n",NoClasses) ;
       for (i=0 ; i < NoClasses ; i++)
         fprintf(fout, "%d\n", i);
      }
      else 
         fprintf(fout,"0 0.0 0.0\n") ;
  }

  for (j=0 ; j < NoRndAttr ; j++)
  {
     fprintf(fout,"R%d\n",j+1) ;
     if (AttrMode)
     {
       fprintf(fout,"%d\n",NoClasses) ;
       for (i=0 ; i < NoClasses ; i++)
         fprintf(fout, "%d\n", i);
      }
      else 
         fprintf(fout,"0 0.0 0.0\n") ;
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

  // print out class distribution
  printf("Class value's probabilities\n") ;
  for (j=1 ; j <= NoClasses ; j++)
      printf("%d=%d(%.4f)  ",j,classProb[j-1],classProb[j-1]/double(NumberOfExamples)) ;
  printf("\n");

   return 0 ;

}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "contain.h"

inline double randBetween(double From, double To)
{
   return From + (double(rand())/RAND_MAX) * (To-From) ;
}

int rndModuloNot(int modulo, int not)
{
   int rnd = rand() % modulo ;
   if (rnd != not)
       return rnd ;
   else return rndModuloNot(modulo, not) ;
}

int rndModuloP(int modulo, marray<double> &pSum)
{
    double x = randBetween(0, 1.0) ;
    for (int i= 0 ; i < modulo ; i++)
        if (x <= pSum[i])
            return i ;
    return -1 ;
}

const double eps = 1e-7 ;

main(int argc, char *argv[])
{
  int noMustArgs = 4 ;
  if (argc < noMustArgs+1)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutFileName NoClasses NoRandomAttr NoExamples ClassProbs\n",argv[0]) ;
     exit(1) ;
  }
  char buf[MaxPath] ;
  sprintf(buf,"%s.data",argv[1]) ;
  FILE *fout ;
  if ((fout = fopen(buf,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }
  int i, j, k ;
  int noClasses = atoi(argv[2]) ; ;
  int modulo = 2;  
  int noRndAttr = atoi(argv[3]) ; 
  int noExamples = atoi(argv[4]) ;
  if (noExamples <= 0)
    goto wrongCall ;
  marray<double> cProb(noClasses+1,0.0);
  double cSum = 0.0 ;
  if (argc != noMustArgs + 1+ noClasses)
     goto wrongCall ;
  for (i=1 ; i <= noClasses; i++) {
     cProb[i] = atof(argv[noMustArgs+i]) ;
     if (cProb[i] < eps)
        cProb[i] = eps ;
     cSum += cProb[i] ;
  }
  if (cSum < 0.999 || cSum > 1.001)  {
      fprintf(stderr, "Error: probabilities does not sum to 1.\n") ;
      exit(1) ;
  }
  srand( (unsigned)time( NULL ) );

  // generate description in C4.5 format
  sprintf(buf,"%s.names",argv[1]) ;
  if ((fout = fopen(buf,"w")) == NULL) {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }
  int noCorr = 2 ;
  marray<int> corr(noCorr) ;
  corr[0] = 90 ;  corr[1] = 70 ;
  // corr[2] = 65 ;  corr[3] = 50 ;
  
  fprintf(fout, "|Cost sensitive problem with %d classes, XOR informative and %d random attributes\n", noClasses, noRndAttr) ;
  fprintf(fout,"C%d.\n",noClasses) ;
  fprintf(fout,"C%d: 0 ", noClasses) ; 
  for (i=1 ; i < noClasses ; i++) 
     fprintf(fout,", %d", i) ; 
  fprintf(fout,".\n") ;
  for (i=0 ; i < noClasses ; i++) {
     for (k=0 ; k < noCorr ; k++) {
        fprintf(fout,"A-%d-%d: 0", i,corr[k]) ;
        for (j=1 ; j< modulo ; j++)
          fprintf(fout, ", %d", j) ;
        fprintf(fout, ".\n") ;
        fprintf(fout,"X1-%d-%d: 0", i,corr[k]) ;
        for (j=1 ; j< modulo ; j++)
          fprintf(fout, ", %d", j) ;
        fprintf(fout, ".\n") ;
        fprintf(fout,"X2-%d-%d: 0", i,corr[k]) ;
        for (j=1 ; j< modulo ; j++)
          fprintf(fout, ", %d", j) ;
        fprintf(fout, ".\n") ;
     }
  }
  for (k=0 ; k < noRndAttr ; k++) {
      fprintf(fout,"R-%d: 0",int((0.5+k/2.0/noRndAttr)*100)) ;
      for (j=1 ; j< modulo ; j++)
         fprintf(fout, ", %d", j) ;
      fprintf(fout, ".\n") ;
  }
  fclose(fout) ;

  // generate data
  sprintf(buf,"%s.data",argv[1]) ;
  if ((fout = fopen(buf,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }
  // a comment line with attribute names
  fprintf(fout,"| C%d", noClasses) ;
  for (i=0 ; i < noClasses ; i++) {
     for (k=0 ; k < noCorr ; k++) {
        fprintf(fout,",A-%d-%d", i,corr[k]) ;
        fprintf(fout,",X1-%d-%d", i,corr[k]) ;
        fprintf(fout,",X2-%d-%d", i,corr[k]) ;
     }
  }
  for (k=0 ; k < noRndAttr ; k++) 
      fprintf(fout,",R-%d",int((0.5+k/2.0/noRndAttr)*100)) ;
  fprintf(fout,"\n") ;

  // class probabilities
  marray<double> cPsum(noClasses+1) ;
  cPsum[0] = 0.0 ;
  for (i=1 ; i <= noClasses; i++)
      cPsum[i] = cPsum[i-1] + cProb[i] ;
  
  // generate labels
  marray<int> dFunction(noExamples, 0) ;
  for (i=0 ; i < noClasses ; i++)
      for(j=int(cPsum[i] * noExamples) ; j < cPsum[i+1] * noExamples ; j++)
          dFunction[j] = i ;
  // scramble labels
  for (i=0 ; i< noExamples-1 ; i++) 
      swap(dFunction[i], dFunction[i+rand()%(noExamples-i)]) ;
  int a, x1, x2 ;
  for (i=0; i < noExamples ; i++)
  {
     fprintf(fout,"%d",dFunction[i]) ;
     for (k=0 ; k < noClasses ; k++) {   
        for (j=0 ; j < noCorr ; j++) {
           if  (dFunction[i] == k) // generate correlation for that class, other are random 
              a = (randBetween(0, 1.0) <= corr[j]/100.0 ? 1 : 0) ;
           else a = rand() % modulo ;
           x1 = rand() % modulo ;
           x2 = (a - x1 + modulo) % modulo ;
           fprintf(fout,",%d,%d,%d",a,x1,x2) ;
        }
     }
     for (j=0 ; j < noRndAttr ; j++) 
      if (randBetween(0,1.0) < 0.5+j/2.0/noRndAttr) 
        fprintf(fout,",0") ;
      else
        fprintf(fout,",%d", 1+rand() % (modulo-1)) ;

     if (i < noExamples -1)
         fprintf(fout,"\n") ;     
  }
  fclose(fout) ;

   return 0 ;

}
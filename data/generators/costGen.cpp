#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "contain.h"

inline double randBetween(double From, double To)
{
   return From + (double(rand())/RAND_MAX) * (To-From) ;
}

const double eps = 1e-7 ;

main(int argc, char *argv[])
{
  int NoMustArgs = 7 ;
  if (argc < NoMustArgs+1)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutFileName NoClasses TypeOfCost HitLow HitHigh MissLow MissHigh [ClassProbabilities] \n",argv[0]) ;
     fprintf(stderr,"\t TypeOfCost: 1-random, 2-irrespectable of predicted class  C(i,j)=C(j), 3 - reverse proportionaly to class probabilities\n") ; 
     fprintf(stderr,"\t             4-C(i,i)=0, economy setting, class value is reverse proportionaly to class probabilities, unit transaction cost\n") ; 
     fprintf(stderr,"\t             5-C(i,i)=0, economy setting, class value is random, unit transaction cost\n") ; 
     exit(1) ;
  }
  FILE *fout ;
  if ((fout = fopen(argv[1],"w")) == NULL)
  {
     error("Cannot write to file ",argv[1]) ;
     exit(1) ;
  }

  int NoClasses = atoi(argv[2]) ;
  if (NoClasses <= 0)
    goto wrongCall ;
  int TypeOfCost = atoi(argv[3]) ;
  if (TypeOfCost < 0 || TypeOfCost > 5)
    goto wrongCall ;
  double HitLow = atof(argv[4]) ;
  double HitHigh = atof(argv[5]) ;
  double MissLow = atof(argv[6]) ;
  double MissHigh = atof(argv[7]) ;
  marray<double> cProb(NoClasses+1,0.0);
  double cSum = 0.0 ;
  if (TypeOfCost == 3 || TypeOfCost == 4) {
      if (argc != NoMustArgs + 1+ NoClasses)
          goto wrongCall ;
      for (int i=1 ; i <= NoClasses; i++)  {
          cProb[i] = atof(argv[NoMustArgs+i]) ;
          if (cProb[i] < eps)
              cProb[i] = eps ;
          cSum += cProb[i] ;
      }
      if (cSum < 0.999 || cSum > 1.001) {
        fprintf(stderr, "Error: probabilities does not sum to 1.\n") ;
        exit(1) ;
      }
  }

  srand((unsigned)time(NULL)) ;
  mmatrix<double> Cost(NoClasses+1,NoClasses+1,0.0) ;
 
  int cT,cP ;
  double randomMiss ;
  for (cT= 1; cT <= NoClasses ; cT++)
  {
     Cost(cT,cT) = randBetween(HitLow, HitHigh)  ;
     if (TypeOfCost ==2)
         randomMiss = randBetween(MissLow, MissHigh)  ;
     for (cP=1 ; cP <= NoClasses; cP++)
         switch (TypeOfCost) {
         case 1: if (cP ==cT) 
                     Cost(cP,cT) = randBetween(HitLow, HitHigh)  ;
                 else  
                     Cost(cP,cT) = randBetween(MissLow, MissHigh)  ;
                 break ;
         case 2: if (cP ==cT) 
                     Cost(cP,cT) = randBetween(HitLow, HitHigh)  ;
                 else  
                   Cost(cP,cT) = randomMiss ;
                 break ;
         case 3: if (cP ==cT) 
                   Cost(cP,cT) = randBetween(HitLow, HitHigh)  ;
                 else  
                   Cost(cP,cT) =  randBetween(MissLow, MissHigh*cProb[cP]/cProb[cT]);
                 break ;
         case 4: 
         case 5: if (cP ==cT) 
                   Cost(cP,cT) = 0 ;
                 else if (value[cP] > value[cT])  
                     Cost(cP,cT) =  transactionCost ;
                 else Cost(cP,cT) =  value[cT] - value[cP] ;
                 break ;
         }
  }

  for (cT=1 ; cT <= NoClasses ; cT++) {
      for (cP=1 ; cP<=NoClasses ; cP++)
        fprintf(fout,"%10.4f  ",Cost(cP,cT)) ;
      fprintf(fout,"\n") ;
  }

  fclose(fout) ;

  return 0 ;
}
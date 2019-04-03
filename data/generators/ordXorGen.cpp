#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include "contain.h"
#include "ordEvalGen.h"

inline double randBetween(double From, double To)
{
   return From + (double(rand())/RAND_MAX) * (To-From) ;
}


int main(int argc, char *argv[])
{
  if (argc != 5)
  {
   wrongCall:
     fprintf(stderr, "\nUsage: %s OutDomainName CombinationLevel NumberOfRandomAttr NumberOfExamples\n", argv[0]) ;
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
  
  int combinationLevel = atoi(argv[2]) ;
  int NoRandomAttr = atoi(argv[3]) ;
  if (NoRandomAttr < 0)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[4]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
  //double PercentNoise = atof(argv[4]) ; 
  //if (PercentNoise < 0 || PercentNoise > 100)
  //  goto wrongCall ;
    
  srand((unsigned)time(NULL)) ;

  int NoInfAttr = 3;
 
  int noDiscrete = combinationLevel*NoInfAttr+NoRandomAttr ;
  mmatrix<double> D(NumberOfExamples, noDiscrete+1);
  marray<double> valArray(NumberOfExamples) ;
  int noValues = 5 ;


  fprintf(fout, "%d\n",NumberOfExamples) ;
  int i, j,k;
  double minF = FLT_MAX, maxF=-FLT_MAX, basic, performance, excitement ;
  for (i=0; i < NumberOfExamples ; i++)
  {
     for (j = 1 ;  j <=noDiscrete; j++)
       D(i,j) = 1.0 + rand() % noValues ;

     D(i,0) = 0.0 ;
	 // basic attributes
	 for (basic=0, j=1 ; j <= combinationLevel ; basic+=D(i,j), j++) ;
	 for (performance=0, j=1+combinationLevel ; j <= 2*combinationLevel ; performance+=D(i,j), j++) ;
	 for (excitement=0, j=1+2*combinationLevel ; j <= 3*combinationLevel ; excitement+=D(i,j), j++) ;
	 D(i,0) += basicAttr(basic/double(combinationLevel), 1.0) ; 
	 // performance
	 D(i,0) += performanceAttr(performance/double(combinationLevel), 1.0); 
	 // excitment
	 D(i,0) += excitementAttr(excitement/double(combinationLevel), 1.0); 

	 valArray[i] = D(i,0) ;
	 if ( D(i,0) < minF )
	 	 minF = D(i,0) ;
	 if ( D(i,0) > maxF )
	 	 maxF = D(i,0) ;
  }
  valArray.setFilled(NumberOfExamples) ;
  valArray.qsortAsc(); 
  double rangeF = maxF - minF ;
  minF += 0.00001 ;

  for (i=0; i < NumberOfExamples ; i++)  {
     D(i,0) = 1.0+int((D(i,0)-minF)/rangeF*double(noValues)) ;

	 fprintf(fout,"%4d   ", int(D(i,0))) ;
     for (j=1 ; j <= noDiscrete ; j++)
   	   fprintf(fout,"%4d ",int(D(i,j))) ;
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

   fprintf(fout,"%d\nSatisfaction\n%d\n", noDiscrete+1, noValues) ;
   for (k=1 ;  k<= noValues ; k++)
	   fprintf(fout, "%d\n", k) ;

   for (j=0 ; j < NoInfAttr*combinationLevel ; j++)   {
	   switch (j/combinationLevel) {
		   case 0:fprintf(fout,"Basic") ; break;
		   case 1:fprintf(fout,"Performance") ; break ;
  		   case 2:fprintf(fout,"Excitement") ; break; 
	   }
	   /*switch (j%combinationLevel) {
		   case 0:fprintf(fout,"90") ; break ;
		   case 1:fprintf(fout,"60") ; break ;
  		   case 2:fprintf(fout,"30") ; break ;
	   }
	   */
	   fprintf(fout,"%d\n%d\n",1+j%combinationLevel,noValues) ;
       for (k=1 ; k <= noValues ; k++)
         fprintf(fout,"%d\n",k) ;

   }
   for (j=0 ; j < NoRandomAttr ; j++)   {
	   fprintf(fout,"Random%d\n%d\n",j+1, noValues) ;
       for (k=1 ; k <= noValues ; k++)
         fprintf(fout,"%d\n",k) ;
   }

    fclose(fout) ;

  // generate split file: all examples are for training
  /*
  sprintf(fileName,"%s.99s",argv[1]) ;
  if ((fout = fopen(fileName,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",fileName) ;
     return 1 ;
  }
  for (i=0; i < NumberOfExamples ; i++)
    fprintf(fout,"0\n") ;
  fclose(fout) ;
  */
  return 0 ;

}


double basicAttr(double value, double prob) {
	if (randBetween(0.0,1.0) < prob) { 
		if (value <= 3.0)
			return -1.0 ;
		else
			return 1.0 ;
	}
	else return 0.0 ;
}

double performanceAttr(double value, double prob) {
	if (randBetween(0.0,1.0) < prob) 
		return value -2.0 ;
	else return 0.0 ;
}

double excitementAttr(double value, double prob) {
	if (randBetween(0.0,1.0) < prob) {
		if (value == 5 )
			return 2.0 ;
		else
			return 0.0 ;
	}
	else return 0.0 ;

}


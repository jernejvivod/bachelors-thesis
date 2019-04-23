//
// See Simulation, Modelling & Analysis by Law & Kelton, pp259
//
// This is the ``polar'' method.
//

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "contain.h"


union PrivateRNGDoubleType {                      // used to access doubles as unsigneds
    double d;
    unsigned int u[2];
};


boolean haveCachedNormal = FALSE ;
double cachedNormal ;
double pStdDev = 1.0 ;
double pMean = 0.0 ;


PrivateRNGDoubleType doubleMantissa;


unsigned int asLong(void)
{
   unsigned int temp ;
   temp = rand() * 32768 + rand() ;
   return temp ;
}


double RNGasDouble()
{
    PrivateRNGDoubleType result;
    result.d = 1.0;
    result.u[0] |= (asLong() & doubleMantissa.u[0]);
    result.u[1] |= (asLong() & doubleMantissa.u[1]);
    result.d -= 1.0;
    assert( result.d < 1.0 && result.d >= 0);
    return( result.d );
}


void RNGinit(void)
{
    static char initialized = 0;

    if (!initialized)
   {

   assert (sizeof(double) == 2 * sizeof(unsigned int));

   //
   // The following is a hack that I attribute to
   // Andres Nowatzyk at CMU. The intent of the loop
   // is to form the smallest number 0 <= x < 1.0,
   // which is then used as a mask for two longwords.
   // this gives us a fast way way to produce double
   // precision numbers from longwords.
   //
   // I know that this works for IEEE and VAX floating
   // point representations.
   //
   // A further complication is that gnu C will blow
   // the following loop, unless compiled with -ffloat-store,
   // because it uses extended representations for some of
   // of the comparisons. Thus, we have the following hack.
   // If we could specify #pragma optimize, we wouldn't need this.
   //

   PrivateRNGDoubleType t;


   t.d = 1.5;
   if ( t.u[1] == 0 ) {                     // sun word order?
       t.u[0] = 0x3fffffff;
       t.u[1] = 0xffffffff;
   }
   else {
       t.u[0] = 0xffffffff;              // encore word order?
       t.u[1] = 0x3fffffff;
   }


   // set doubleMantissa to 1 for each doubleMantissa bit
   doubleMantissa.d = 1.0;
   doubleMantissa.u[0] ^= t.u[0];
   doubleMantissa.u[1] ^= t.u[1];

   initialized = 1;
    }

}




double Normal(void)
{

    if (haveCachedNormal)
    {
      haveCachedNormal = FALSE;
      return (cachedNormal * pStdDev + pMean );
    }
    else
    {
      for(;;)
      {
        double u1 = RNGasDouble();
        double u2 = RNGasDouble();
        double v1 = 2 * u1 - 1;
        double v2 = 2 * u2 - 1;
        double w = (v1 * v1) + (v2 * v2);

//
// We actually generate two IID normal distribution variables.
// We cache the one & return the other.
//
        if (w <= 1)
        {
          double y = sqrt( (-2 * log(w)) / w);
          double x1 = v1 * y;
          double x2 = v2 * y;

          haveCachedNormal = TRUE;
          cachedNormal = x2;
          return (x1 * pStdDev + pMean);
       }
     }
    }
}



inline float randBetween(float From, float To)
{
   return From + (float(rand())/RAND_MAX) * (To - From) ;
}



inline double sqr(double x)
{
   return x*x ;
}


int main(int argc, char *argv[])
{

  if (argc != 3)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutFileName NumberOfExamples",argv[0]) ;
     return 1 ;
  }
  FILE *fout ;
  if ((fout = fopen(argv[1],"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }

  int NumberOfExamples = atoi(argv[2]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;

  srand((unsigned)time(NULL)) ;
  RNGinit() ;


  fprintf(fout, "%d\n",NumberOfExamples) ;
  int j ;


  marray<double> noise(NumberOfExamples) ;

  pStdDev = sqrt(2.0) ;
  pMean = 0 ;
  double avg = 0.0 , std = 0.0, SE=0.0 ;
  for (int i=0 ; i < NumberOfExamples ; i++)
  {
    noise[i] = Normal() ;
    avg += noise[i] ;
    SE += sqr(noise[i]) ;

   }
   avg /= double(NumberOfExamples) ;
   SE /= double(NumberOfExamples) ; 
  for (i=0 ; i < NumberOfExamples ; i++)
  {
     std += sqr(noise[i] - avg) ;
  }
  std = sqrt(std/double(NumberOfExamples)) ;


  printf("theoretical mu:%8.3f theoretical variance: %8.3f\n",0.0, 2.0) ;
  printf("       real mu:%8.3f        real variance: %8.3f\n",avg, sqr(std)) ;

  int NoContinuous = 10 ;
  marray<double> C(NoContinuous) ;
  marray<double> fValue(NumberOfExamples) ;
  double dtemp, Function ;
  int firstBranch = 0;
  double functionMean = 0.0 ;

  for (i=0; i < NumberOfExamples ; i++)
  {
      if (randBetween(0.0, 1.0) <= 0.5)
         C[0] = 1.0 ;
      else 
        C[0] = -1.0 ;


      for (j=1 ; j < NoContinuous ; j++)
      {
         C[j] = int(randBetween(0.0, 3.0)) - 1.0 ;

      }
      if (C[0] == 1.0)
      {
        Function = 3.0 + 3.0*C[1] + 2.0*C[2] + C[3] + noise[i] ;
        firstBranch++ ; 
      }
      else
        Function = -3.0 + 3.0*C[4] + 2.0*C[5] + C[6] + noise[i] ;
      fValue[i] = Function ; 

      functionMean += Function ;

      fprintf(fout,"%10.4f  ",Function) ;

     for (j=0 ; j < NoContinuous ; j++)
       fprintf(fout,"%10.3f ", C[j]) ;

     fprintf(fout,"\n") ;

  }
  functionMean /= double(NumberOfExamples) ;
  fclose(fout) ;
  double defaultError = 0.0 ;
  for (i=0 ; i < NumberOfExamples ; i++)
     defaultError += sqr(fValue[i] - functionMean) ;
  defaultError /= double(NumberOfExamples) ;

  printf("theoretical first:%d theoretical second: %d\n", NumberOfExamples/2,NumberOfExamples/2 ) ;
  printf("       real first:%d        real second: %d\n\n", firstBranch, NumberOfExamples-firstBranch) ;

  printf(" Optimal classifier's root of squared error: %8.3f\n", sqrt(SE)) ;
  printf(" Default classifier's root of squared error: %8.3f\n", sqrt(defaultError)) ;
  printf("Optimal relative squared error: %8.3f\n", SE/defaultError) ;

  return 0 ;
}
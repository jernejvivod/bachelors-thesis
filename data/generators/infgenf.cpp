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

  int N1 = 500 ;
  int N2 = 500 ;
  int N = N1 + N2 ;
  assert(NumberOfExamples == N) ;

  marray<double> number(N) ;

  // set desired values
  double mu = 0.0 ;
  double mu1 = 1.0 ;
  double mu2 = (mu * N - mu1*N1) / N2 ;

  double s = 1.0 + 1.0 / 9.0 ;
  double s1 = 1.0 / 9.0 ;
  double s2 = ( (s + sqr(mu))*N - (s1 + sqr(mu1))*N1 ) / N2 - sqr(mu2) ;


  // get real values and compute their parameters
  pStdDev = sqrt(s1) ;
  pMean = mu1 ;
  double avg = 0.0 , std = 0.0 ;
  double avg1 = 0.0, std1 = 0.0 ;
  for (int i=0 ; i < N1 ; i++)
  {
    number[i] = Normal() ;
    avg1 += number[i] ;
   }
   avg = avg1 ;
   avg1 /= double(N1) ;

  pStdDev = sqrt(s2) ;
  pMean = mu2 ;
  double avg2 = 0.0, std2 = 0.0 ;
  for (i=N1 ; i < N ; i++)
  {
    number[i] = Normal() ;
    avg2 += number[i] ;
   }
   avg += avg2 ;
   avg /= double(N) ;
   avg2 /= double(N2) ;

  for (i=0 ; i < N1 ; i++)
  {
     std += sqr(number[i] - avg) ;
     std1 += sqr(number[i] - avg1) ;
  }
  std1 = sqrt(std1/double(N1)) ;

  for (i=N1 ; i < N ; i++)
  {
     std += sqr(number[i] - avg) ;
     std2 += sqr(number[i] - avg2) ;
  }
  std2 = sqrt(std2/double(N2)) ;

  std = sqrt(std/double(N)) ;

  printf("                together      left     right\n") ;
  printf("theoretical mu:%8.3f %8.3f %8.3f\n",mu, mu1, mu2) ;
  printf("       real mu:%8.3f %8.3f %8.3f\n",avg, avg1, avg2) ;
  printf("theoretical  s:%8.3f %8.3f %8.3f\n",sqrt(s), sqrt(s1), sqrt(s2)) ;
  printf("       real  s:%8.3f %8.3f %8.3f\n",std, std1, std2) ;

  int NoDiscrete = 10 ;
  int NoContinuous = 10 ;
  marray<int> D(NoDiscrete) ;
  marray<float> C(NoContinuous) ;
  float Function ;

  for (i=0; i < NumberOfExamples ; i++)
  {
      Function = number[i] ;
      if (i < N1)
      {
         C[0] = randBetween(0.0, 0.5) ;
         D[0] = 0 ;
      }
      else
      {
         C[0] = randBetween(0.5, 1.0) ;
         D[0] = 1 ;
      }

      D[1] = rand() % 2 ;
      D[2] = abs(D[0]-D[1]) % 2 ;

      D[3] = rand() % 2 ;
      D[4] = rand() % 2 ;
      D[5] = abs(D[0]-D[3]-D[4]) % 2 ;

      for (j=6 ; j < NoDiscrete ; j++)
        D[j] = rand() %2 ;

      C[1] = randBetween(0.0, 1.0) ;
      C[2] = C[0] - C[1] ;
      C[2] += fabs(floor(C[2])) ;

      C[3] = randBetween(0.0, 1.0) ;
      C[4] = randBetween(0.0, 1.0) ;
      C[5] = C[0] - C[3] - C[4] ;
      C[5] += fabs(floor(C[5])) ;

      for (j=6 ; j < NoContinuous ; j++)
        C[j] = randBetween(0.0, 1.0) ;


     fprintf(fout,"%10.3f  ",Function) ;

     for (j=0 ; j < NoDiscrete ; j++)
       fprintf(fout,"%4d ",int(D[j]+1)) ;

     for (j=0 ; j < NoContinuous ; j++)
       fprintf(fout,"%10.3f ", C[j]) ;

     fprintf(fout,"\n") ;

  }
  fclose(fout) ;

   return 0 ;
}

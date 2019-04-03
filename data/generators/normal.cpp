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


booleanT haveCachedNormal = mFALSE ;
double cachedNormal ;
//double pStdDev = 1.0 ;
//double pMean = 0.0 ;


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




double Normal(double pMean, double pStdDev)
{

    if (haveCachedNormal)
    {
      haveCachedNormal = mFALSE;
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

          haveCachedNormal = mTRUE;
          cachedNormal = x2;
          return (x1 * pStdDev + pMean);
       }
     }
    }
}


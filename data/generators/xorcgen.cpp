#include <fstream.h>

const int base = 10;

main()
{
  ofstream to("par3-10.dat") ;
  int table[base];
  for (int i=0; i<1024 ; i++)
  {
     //  x1   x2   x3  x4  x5  x6  x7  x8  x9 x10
     // 512  256  128  64  32  16   8   4   2   1
     if ( ( (i&512) && (i&128) && (i&64) ) ||
          ( (i&512) && (!(i&128)) && (!(i&64)) ) ||
          ( (!(i&512)) && (i&128) && (!(i&64)) ) ||
          ( (!(i&512)) && (!(i&128)) && (i&64) )  )
     {
          to << "1   " ;
     }
     else to << "0   " ;
     int divider = 512 ;
     int j =  i;
     while (divider)
     {
        to <<  j/divider  << " " ;
        j %= divider;
        divider /= 2 ;
     }
     to << "\n" ;
  }
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "contain.h"



int main(int argc, char *argv[])
{
  if (argc != 7)
  {
   wrongCall:
     fprintf(stderr, "Usage: %s OutFileName NoInfDisc NoRandDisc NoDiscValues NoInfCont NoRandCont",argv[0]) ;
     return 1 ;
  }
  FILE *fout ;
  if ((fout = fopen(argv[1],"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",argv[1]) ;
     return 1 ;
  }

  int NoInfDisc = atoi(argv[2]) ;
  if (NoInfDisc < 0)
    goto wrongCall ;
  int NoRandDisc = atoi(argv[3]) ;
  if (NoRandDisc < 0)
    goto wrongCall ;
  int NoDiscValues = atoi(argv[4]) ;
  if (NoRandDisc < 0)
    goto wrongCall ;
  int NoInfCont = atoi(argv[5]) ;
  if (NoInfCont < 0)
    goto wrongCall ;
  int NoRandCont = atoi(argv[6]) ;
  if (NoRandCont < 0)
    goto wrongCall ;
  
  
  
  fprintf(fout, "%d \n",NoInfDisc+NoRandDisc+NoInfCont+NoRandCont+1) ;
  fprintf(fout, "F \n0 0.0 0.0 \n") ;
  int i, j ;
  for (i=1; i <= NoInfDisc ; i++)
  {
     fprintf(fout, "Id%d \n%d \n",i,NoDiscValues) ;
     for (j=1 ; j <= NoDiscValues ; j++)
       fprintf(fout, "%d \n", j) ;
  }
  for (i=1; i <= NoRandDisc ; i++)
  {
     fprintf(fout, "Rd%d \n%d \n",i,NoDiscValues) ;
     for (j=1 ; j <= NoDiscValues ; j++)
       fprintf(fout, "%d \n", j) ;
  }
  for (i=1; i <= NoInfCont ; i++)
  {
     fprintf(fout, "Ic%d \n0 0.0 0.0 \n",i) ;
  }
  for (i=1; i <= NoRandCont ; i++)
  {
     fprintf(fout, "Rc%d \n0 0.0 0.0 \n",i) ;
  }
  
  fclose(fout) ;

  return 0 ;

}
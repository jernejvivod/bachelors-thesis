
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "contain.h"


int main(int argc,char *argv[])
{
  if (argc!=5)
  {
wrongCall:
     fprintf(stderr,"Usage : %s DomainName NumberOfTrain NumberOfCases NoFiles", argv[0] );
     return 1 ;
  }
  FILE *fout ;

  const int MaxPath = 512 ;
  char path[MaxPath] ;

  srand((unsigned)time(NULL)) ;

  int NoTrain = atoi(argv[2]) ;
  int NoCases = atoi(argv[3]) ;
  if (NoTrain >= NoCases) 
     goto wrongCall ;
  int NoFiles = atoi(argv[4]) ;
  
  srand((unsigned)time(NULL)) ;

  marray<int> outTable(NoCases) ;
  int i, j,selected ;

   for (j=0; j<NoFiles ; j++)
  {
     sprintf(path,"%s.%02ds", argv[1],j) ;
     fout = fopen(path,"w") ;

     outTable.init(1) ;

     for (i=0 ; i < NoTrain ; i++)
     { 
        do {
          selected = rand() % NoCases ;
        } while (outTable[selected] == 0) ;

        outTable[selected] = 0 ;
      }

      for(i=0 ; i < NoCases ; i++)
      	 fprintf(fout,"%d\n",outTable[i]) ;
       
     fclose(fout) ;
  }

   return 0 ;
}

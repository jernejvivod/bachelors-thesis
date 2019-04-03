
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "contain.h"


int main(int argc,char *argv[])
{
  if (argc!=4)
  {
     fprintf(stderr,"Usage : %s DomainName NoSplits NumberOfCases", argv[0] );
     return 1 ;
  }
  FILE *fout ;

  const int MaxPath = 512 ;
  char path[MaxPath] ;

  srand((unsigned)time(NULL)) ;

  int NumberOfFiles = atoi(argv[2]) ;
  int NoCases = atoi(argv[3]) ;

  srand((unsigned)time(NULL)) ;

  marray<int> splitTable(NoCases), outTable(NoCases) ;
  for (int i=0 ; i < NoCases; i++)
    splitTable[i] = i ;

  int selected ;
  int upper = NoCases ;
  for (i=0 ; i < NoCases ; i++)
  {
      selected = rand() % upper ;
      outTable[i] = splitTable[selected] ;
      splitTable[selected] = splitTable[--upper] ;
  }

  // determine how many files define split with one element more than NoCases/NumberOfFiles
  upper = NoCases % NumberOfFiles ;
  int noElem = NoCases / NumberOfFiles ;
  for (i=0; i<upper ; i++)
  {
     sprintf(path,"%s.%02ds", argv[1],i) ;
     fout = fopen(path,"w") ;

     for (int j=0 ; j < NoCases ; j++)
       if (outTable[j] >= i*(noElem+1) && outTable[j] < (i+1)*(noElem+1) )
     	 fprintf(fout,"0\n") ;
       else
	     fprintf(fout,"1\n") ;

     fclose(fout) ;
  }

  int alreadyUsed = upper * (noElem+1) ;
  // splits with NoCases/NumberOfFiles
  for (i=upper; i<NumberOfFiles ; i++)
  {
     sprintf(path,"%s.%02ds", argv[1],i) ;
     fout = fopen(path,"w") ;

     for (int j=0 ; j < NoCases ; j++)
       if (outTable[j] >= alreadyUsed + (i-upper)*noElem &&
           outTable[j] < alreadyUsed +(i+1-upper)*noElem )
	      fprintf(fout,"0\n") ;
       else
	      fprintf(fout,"1\n") ;

     fclose(fout) ;
  }


  return 0 ;
}

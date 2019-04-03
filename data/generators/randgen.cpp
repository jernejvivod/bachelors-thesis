#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main(int argc,char *argv[])
{
  if (argc!=4)
  {
     fprintf(stderr,"Usage : %s DomainName NumberOfFiles NumberOfCases", argv[0] );
     return 1 ;
  }
  FILE *fout ;

  const int MaxPath = 512 ;
  char path[MaxPath], buf[MaxPath] ;

  int NumberOfFiles = atoi(argv[2]) ;
  int NoCases = atoi(argv[3]) ;

  srand((unsigned)time(NULL)) ;

  for (int i=0; i<NumberOfFiles ; i++)
  {
     sprintf(path,"%s.%02ds", argv[1],i) ;
     fout = fopen(path,"w") ;

     for (int j=0 ; j < NoCases ; j++)
       if (rand()%100<30)
	 fprintf(fout,"1\n") ;
       else
	 fprintf(fout,"0\n") ;

     fclose(fout) ;
  }
  return 0 ;
}
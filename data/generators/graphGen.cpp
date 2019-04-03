#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
    if (argc != 3){
        fprintf(stderr,"Usage: %s SteviloTock ImeDatoteke",argv[0])   ;
        exit(1) ;
    }
   int tock = atoi(argv[1]) ;

   FILE *fout=fopen(argv[2],"w") ;
   int i,j ;
   for (i=1 ; i <= tock ; i++)
       for  (j=1 ; j <= tock; j++)
          if (rand()%100 < 75)
              fprintf(fout,"u(v%d, v%d, %d).\n",i,j,rand()%10000) ;
   fclose(fout) ;
}
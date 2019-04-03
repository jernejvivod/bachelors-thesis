#include <stdlib.h>
#include <fstream.h>
#include <iostream.h>

int main(int argc,char *argv[])
{
  if (argc!=4)
  {
     cerr << "Usage : xorc NumberOfCases OutFile Seed" ;
     return 1 ;
  }
  ofstream to(argv[2]) ;
  int max = atoi(argv[1]) ;
  srand(atoi(argv[3]));
  int count[16], j;
  for (int i=0 ; i<15 ; i++)
    count[i] = 0 ;
  for (i=0; i<max ; i++)
  {
      switch ((rand() + rand()) % 16)
      {
	 case 0: to << "2  1  7 " ;
		 count[0] ++ ;
		 break ;
	 case 1: to << "2  2  7 " ;
		 count[1] ++ ;
		 break ;
	 case 2: to << "2  1  6 " ;
		 count[2] ++ ;
		 break ;
	 case 3: to << "2  2  6 " ;
		 count[3] ++ ;
		 break ;
	 case 4: to << "1  2  2 " ;
		 count[4] ++ ;
		 break ;
	 case 5: to << "1  3  2 " ;
		 count[5] ++ ;
		 break ;
	 case 6: to << "1  2  1 " ;
		 count[6] ++ ;
		 break ;
	 case 7: to << "1  3  1 " ;
		 count[7] ++ ;
		 break ;
	 case 8: to << "2  5  3 " ;
		 count[8] ++ ;
		 break ;
	 case 9: to << "2  6  3 " ;
		 count[9] ++ ;
		 break ;
	 case 10: to << "2  5  2 " ;
		 count[10] ++ ;
		 break ;
	 case 11: to << "2  6  2 " ;
		 count[11] ++ ;
		 break ;
	 case 12: to << "1  6  6 " ;
		 count[12] ++ ;
		 break ;
	 case 13: to << "1  7  6 " ;
		 count[13] ++ ;
		 break ;
	 case 14: to << "1  6  5 " ;
		 count[14] ++ ;
		 break ;
	 case 15: to << "1  7  5 " ;
		 count[15] ++ ;
		 break ;
      }
      for (j = 0 ; j < 10 ; j++)
	if (rand() % 100 < 50)
	   to << "1 " ;
	else
	   to << "2 " ;
      to << "\n" ;
  }
  cout << "Cases : \n"  ;
  for (i=0 ; i < 16 ; i++)
     cout << i << "   " << count[i] <<"\n" ;

  return 0 ;
}
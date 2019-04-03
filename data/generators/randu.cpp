#include <stdio.h>
#define _CRT_RAND_S
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include "contain.h"
#include "utils.h"

//  ********  randu  **********
static unsigned __int64 seedRandu=1; 
void randuSeed(int seed) {
	seedRandu = seed ;
}

void randuOddSeed(int seed) {
	if (seed % 2 == 0) // even
	  seedRandu = seed+1 ;
	else
	  seedRandu = seed ;
}

/* n_i=65539*n_(i-1) mod 2**31 */
int randu() { 
	seedRandu = (65539*seedRandu) & 0x7FFFFFFF; 
    return (int)seedRandu ;
}   
double randuBetween(double From, double To) {
    return From + (randu()/double(0x7FFFFFFF)) * (To-From) ;
}

//  ******** rand() ********************
double msRandBetween(double From, double To) {
    return From + (rand()/double(RAND_MAX)) * (To-From) ;
}

// ********  rand_s  ******************
double msRandSecureBetween(double From, double To) {
    unsigned int number;
    rand_s(&number);
    return From + ((double)number /(double) UINT_MAX) * (To-From) ;
}

// ********  minstd  ******************* 
static int seedMinstd = 1 ;
void minstdSeed(int seed) {
	if (seed < 0)
		seed = -seed ;
	if (seed > 0)
		seedMinstd = seed ;
	else seedMinstd = 1 ;
}

double minstdBetween(double From, double To) {
// Integer Version 2 from Park88 
  const int a = 16807, m = 2147483647, q = 127773 /* m div a */, r = 2836; /* m mod a */
  int  lo, hi, test ;
  hi = seedMinstd / q;
  lo = seedMinstd % q;
  test = a * lo - r * hi;
  if (test > 0)
     seedMinstd = test ;
  else
    seedMinstd = test + m;
  return From + double(seedMinstd) / m * (To-From);
}


// ******** mrg32k5a from L'Ecuyer99  **********
//######################
// incorporated into utils.cpp

// ********** rand_state from Matlab  *******************
#define e53 9007199254740992.0

struct state {
	double z[32],b;
	unsigned int i,j;
} st;

double em53 = 1.0 / e53;
 

union number {
	double a;
	unsigned int b[2];  // b[1] & 0xfffff = bits 51..32, b[0] = bits 31..0 
	// mantissa bits 51..0, see http://en.wikipedia.org/wiki/Double_precision
} p;

void randint()
{
	st.j ^= (st.j<<13);
	st.j ^= (st.j>>17);
	st.j ^= (st.j<<5);
}

void randsetup()
{
	int d,k;
	double x;
	for (k=0; k<32; k++) {
		x = 0;
		for (d=0; d<53; d++) {
			randint();
			x = 2*x + ((st.j>>19)&1);
		}
		st.z[k] = x*em53;
	}
}

double randbits(double x)
{
	unsigned int jhi,jlo;
	jlo = st.j;
	randint();
	jhi = st.j & 0xfffff;
	p.a = x;
	p.b[1] ^= jhi;
	p.b[0] ^= jlo;
	return p.a;
}

void randStateSeed(unsigned int initj)
{
	if (initj == 0) {
		initj = 0x80000000;
	}
	st.j = initj;
	randsetup();
	st.b = 0.0;
	st.i = 0;
	st.j = initj;
}

double get()
{
	double x;
	x = st.z[(st.i+20)&0x1f] - st.z[(st.i+5)&0x1f] - st.b;
	if (x < 0) {
		x += 1.0;
		st.b = em53;
	} else {
		st.b = 0.0;
	}
	st.z[st.i] = x;
	st.i = (st.i+1)&0x1f;
	x = randbits(x);
	return x;
}

double randStateBetween(double From, double To) {
	return From + get() * (To-From) ;
}

// ****** ANSIC
// implementation with 64 bit arithmetic
__int64 seedAnsiC = 1 ;

void ansicSeed(int seed) {
	if (seed < 0)
		seed = -seed ;
	if (seed > 0)
		seedAnsiC = seed ;
	else seedAnsiC = 1 ;
}

double ansicBetween(double From, double To) {
	seedAnsiC = ((seedAnsiC * 1103515245) + 12345) & LONG_MAX;
    return From + double(seedAnsiC)/0x7FFFFFFF * (To-From) ;
}


// ********** WELL19937a
extern "C" void HInitWELLRNG19937a(int nseed, int *vseed);
extern "C" double (*WELLRNG19937a)(void);

void well19937aSeed(int seed) {
	int vseed[1],nseed=1 ;
    vseed[0] = seed ;
	HInitWELLRNG19937a(nseed,vseed);
}


double well19937aBetween(double From, double To) {
    return From + WELLRNG19937a() * (To-From) ;
}



void validate(FILE *fout, int Generator, int Period, int NumberOfExamples) {
  int i,j ;
  srand(7931) ;
  for (i=0; i < NumberOfExamples ; i++)
  {
      for (j=1 ; j <= Period ; j++)
   	    fprintf(fout,"%5d ", rand()) ;
	  fprintf(fout,"\n") ;
  }
} 

int main(int argc, char *argv[])
{
  if (argc != 5)
  {
   wrongCall:
     fprintf(stderr, "\nUsage: %s OutDomainName Generator Period NumberOfExamples", argv[0]) ;
	 fprintf(stderr, "\n          (Generator: 1=RANDU, 2=MSstdlibRand, 3=NRran1, 4=minstd, 5=MRG32k5a, 6=MSstdlibRandSecure,") ;
	 fprintf(stderr, "\n                      7=Matlab rand_state, 8=ANSI C, 9=WELL19937a, 10=RANDUoddSeed)") ;
     return 1 ;
  }
 
  char fileName[MaxFileNameLen] ;
  FILE *fout ;
  
  // open data file
  sprintf(fileName, "%s.data",argv[1]) ;
  if ((fout = fopen(fileName,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",fileName) ;
     return 1 ;
  }
  
  int Generator = atoi(argv[2]) ;
  if (Generator < 1 || Generator >10)
    goto wrongCall ;
  int Period = atoi(argv[3]) ;
  if (Period < 2)
    goto wrongCall ;
  int NumberOfExamples = atoi(argv[4]) ;
  if (NumberOfExamples <= 0)
    goto wrongCall ;
   
  if (strcmp(argv[1], "validate")==0) { 
	  validate(fout, Generator, Period, NumberOfExamples) ;
	  fclose(fout) ;
	  return 0 ;
  }
 
  
  //double nrran1Between(double, double) ; 
  double (*gen)(double, double) ;
  int i, j ;
  switch (Generator) {
  case 1: // randu
	  gen = randuBetween ;
      randuSeed((unsigned)time(NULL)) ;
	  break ;
  case 2: // built in rand()
	  gen = msRandBetween ;
      srand((unsigned)time(NULL)) ; // MS stdlib rand
	  break ;
  case 3:
	  gen = nrran1Between ;
      nrran1Seed((unsigned)time(NULL)) ;
	  break;
  case 4:
	  gen = minstdBetween ;
      minstdSeed((unsigned)time(NULL)) ;
	  break ;
  case 5:
	  gen = mrg32k5aBetween ;
      mrg32k5aSeed((unsigned)time(NULL)) ;
	  break ;
   case 6: // built in rand_s()
	  gen = msRandSecureBetween ;
      // MS stdlib rand_s has no init seed
  	  break ;
   case 7: 
	  gen = randStateBetween ;
      randStateSeed((unsigned)time(NULL)) ; // rand_state
	  break ;
   case 8: 
	  gen = ansicBetween ;
      ansicSeed((unsigned)time(NULL)) ;
	  break ;
   case 9: 
	  gen = well19937aBetween ;
      well19937aSeed((unsigned)time(NULL)) ;
	  break ;
   case 10: 
	  gen = randuBetween ;
	  randuOddSeed((unsigned)time(NULL)) ;
	  break ;

 }
  for (i=0; i < NumberOfExamples ; i++)
  {
      for (j=1 ; j <= Period ; j++)
   	    fprintf(fout,"%e, ",(*gen)(0.0,1.0)) ;
	  fprintf(fout,"\n") ;
  }
  fclose(fout) ;

  // names file
  sprintf(fileName, "%s.names",argv[1]) ;
  if ((fout = fopen(fileName,"w")) == NULL)
  {
     fprintf(stderr,"Cannot write to file %s",fileName) ;
     return 1 ;
  }

  fprintf(fout,"Z%d.\n",Period) ;

  for (j=1 ; j <= Period ; j++)   
	  fprintf(fout,"Z%d:  continuous.\n",j) ;
  fclose(fout) ;

  return 0 ;
}

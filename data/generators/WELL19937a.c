/*********************************************************************************/
/* This file consists of four sections from different authors.                   */
/*                                                                               */
/* The first section is an exact copy of the file WELL19937a.c created            */
/* by F. Panneton, P. L'Ecuyer and M. Matsumoto including the description        */
/* of the file written by its authors.                                           */
/*                                                                               */
/* The second section is copied from the file combmrg2.c created by P. L'Ecuyer  */
/* with a small modification. More detail is given in the description of that    */
/* section, see below.                                                           */
/*                                                                               */
/* The third section is written by P. Savicky and contains the function          */
/* HInitWELLRNG19937a, which is an easier to use replacement of                  */
/* InitWELLRNG19937a. See the DESCRIPTION file for more detail on the            */
/* difference between these two functions.                                       */
/*                                                                               */
/* The last section is written by P. Savicky and contains functions used for     */
/* testing an installation of the previous functions. The test of MRG32k3a uses  */
/* information from the paper by P. L'Ecuyer, where MRG32k3a was presented.      */
/*                                                                               */
/* The original code of the first and second section and the papers describing   */
/* the theory behind them may be found at                                        */
/* http://www.iro.umontreal.ca/~lecuyer/papers.html                              */
/*********************************************************************************/

/* ***************************************************************************** */
/* Copyright:      Francois Panneton and Pierre L'Ecuyer, University of Montreal */
/*                 Makoto Matsumoto, Hiroshima University                        */
/* Notice:         This code can be used freely for personal, academic,          */
/*                 or non-commercial purposes. For commercial purposes,          */
/*                 please contact P. L'Ecuyer at: lecuyer@iro.UMontreal.ca       */
/* ***************************************************************************** */

#define W 32
#define R 624
#define P 31
#define MASKU (0xffffffffU>>(W-P))
#define MASKL (~MASKU)
#define M1 70
#define M2 179
#define M3 449

#define MAT0POS(t,v) (v^(v>>t))
#define MAT0NEG(t,v) (v^(v<<(-(t))))
#define MAT1(v) v
#define MAT3POS(t,v) (v>>t)

/* To obtain the WELL19937c, uncomment the following line */
/* #define TEMPERING                                      */
#define TEMPERB 0xe46e1700U
#define TEMPERC 0x9b868000U

#define V0            STATE[state_i]
#define VM1Over       STATE[state_i+M1-R]
#define VM1           STATE[state_i+M1]
#define VM2Over       STATE[state_i+M2-R]
#define VM2           STATE[state_i+M2]
#define VM3Over       STATE[state_i+M3-R]
#define VM3           STATE[state_i+M3]
#define VRm1          STATE[state_i-1]
#define VRm1Under     STATE[state_i+R-1]
#define VRm2          STATE[state_i-2]
#define VRm2Under     STATE[state_i+R-2]

#define newV0         STATE[state_i-1]
#define newV0Under    STATE[state_i-1+R]
#define newV1         STATE[state_i]
#define newVRm1       STATE[state_i-2]
#define newVRm1Under  STATE[state_i-2+R]

#define FACT 2.32830643653869628906e-10

static int state_i = 0;
static unsigned int STATE[R];
static unsigned int z0, z1, z2;
static double case_1 (void);
static double case_2 (void);
static double case_3 (void);
static double case_4 (void);
static double case_5 (void);
static double case_6 (void);
       double (*WELLRNG19937a) (void);

static unsigned int y;

void InitWELLRNG19937a (unsigned int *init){
   int j;
   state_i = 0;
   WELLRNG19937a = case_1;
   for (j = 0; j < R; j++)
     STATE[j] = init[j];
}

double case_1 (void){
   // state_i == 0
   z0 = (VRm1Under & MASKL) | (VRm2Under & MASKU);
   z1 = MAT0NEG (-25, V0) ^ MAT0POS (27, VM1);
   z2 = MAT3POS (9, VM2) ^ MAT0POS (1, VM3);
   newV1      = z1 ^ z2;
   newV0Under = MAT1 (z0) ^ MAT0NEG (-9, z1) ^ MAT0NEG (-21, z2) ^ MAT0POS (21, newV1);
   state_i = R - 1;
   WELLRNG19937a = case_3;
#ifdef TEMPERING
   y = STATE[state_i] ^ ((STATE[state_i] << 7) & TEMPERB);
   y =              y ^ ((             y << 15) & TEMPERC);
   return ((double) y * FACT);
#else
   return ((double) STATE[state_i] * FACT);
#endif
}

static double case_2 (void){
   // state_i == 1
   z0 = (VRm1 & MASKL) | (VRm2Under & MASKU);
   z1 = MAT0NEG (-25, V0) ^ MAT0POS (27, VM1);
   z2 = MAT3POS (9, VM2) ^ MAT0POS (1, VM3);
   newV1 = z1 ^ z2;
   newV0 = MAT1 (z0) ^ MAT0NEG (-9, z1) ^ MAT0NEG (-21, z2) ^ MAT0POS (21, newV1);
   state_i = 0;
   WELLRNG19937a = case_1;
#ifdef TEMPERING
   y = STATE[state_i] ^ ((STATE[state_i] << 7) & TEMPERB);
   y =              y ^ ((             y << 15) & TEMPERC);
   return ((double) y * FACT);
#else
   return ((double) STATE[state_i] * FACT);
#endif
}

static double case_3 (void){
   // state_i+M1 >= R
   z0 = (VRm1 & MASKL) | (VRm2 & MASKU);
   z1 = MAT0NEG (-25, V0) ^ MAT0POS (27, VM1Over);
   z2 = MAT3POS (9, VM2Over) ^ MAT0POS (1, VM3Over);
   newV1 = z1 ^ z2;
   newV0 = MAT1 (z0) ^ MAT0NEG (-9, z1) ^ MAT0NEG (-21, z2) ^ MAT0POS (21, newV1);
   state_i--;
   if (state_i + M1 < R)
      WELLRNG19937a = case_5;
#ifdef TEMPERING
   y = STATE[state_i] ^ ((STATE[state_i] << 7) & TEMPERB);
   y =              y ^ ((             y << 15) & TEMPERC);
   return ((double) y * FACT);
#else
   return ((double) STATE[state_i] * FACT);
#endif
}

static double case_4 (void){
   // state_i+M3 >= R
   z0 = (VRm1 & MASKL) | (VRm2 & MASKU);
   z1 = MAT0NEG (-25, V0) ^ MAT0POS (27, VM1);
   z2 = MAT3POS (9, VM2) ^ MAT0POS (1, VM3Over);
   newV1 = z1 ^ z2;
   newV0 = MAT1 (z0) ^ MAT0NEG (-9, z1) ^ MAT0NEG (-21, z2) ^ MAT0POS (21, newV1);
   state_i--;
   if (state_i + M3 < R)
      WELLRNG19937a = case_6;
#ifdef TEMPERING
   y = STATE[state_i] ^ ((STATE[state_i] << 7) & TEMPERB);
   y =              y ^ ((             y << 15) & TEMPERC);
   return ((double) y * FACT);
#else
   return ((double) STATE[state_i] * FACT);
#endif
}

static double case_5 (void){
   // state_i+M2 >= R
   z0 = (VRm1 & MASKL) | (VRm2 & MASKU);
   z1 = MAT0NEG (-25, V0) ^ MAT0POS (27, VM1);
   z2 = MAT3POS (9, VM2Over) ^ MAT0POS (1, VM3Over);
   newV1 = z1 ^ z2;
   newV0 = MAT1 (z0) ^ MAT0NEG (-9, z1) ^ MAT0NEG (-21, z2) ^ MAT0POS (21, newV1);
   state_i--;
   if (state_i + M2 < R)
      WELLRNG19937a = case_4;
#ifdef TEMPERING
   y = STATE[state_i] ^ ((STATE[state_i] << 7) & TEMPERB);
   y =              y ^ ((             y << 15) & TEMPERC);
   return ((double) y * FACT);
#else
   return ((double) STATE[state_i] * FACT);
#endif
}

static double case_6 (void){
   // 2 <= state_i <= (R - M3 - 1)
   z0 = (VRm1 & MASKL) | (VRm2 & MASKU);
   z1 = MAT0NEG (-25, V0) ^ MAT0POS (27, VM1);
   z2 = MAT3POS (9, VM2) ^ MAT0POS (1, VM3);
   newV1 = z1 ^ z2;
   newV0 = MAT1 (z0) ^ MAT0NEG (-9, z1) ^ MAT0NEG (-21, z2) ^ MAT0POS (21, newV1);
   state_i--;
   if (state_i == 1)
      WELLRNG19937a = case_2;
#ifdef TEMPERING
   y = STATE[state_i] ^ ((STATE[state_i] << 7) & TEMPERB);
   y =              y ^ ((             y << 15) & TEMPERC);
   return ((double) y * FACT);
#else
   return ((double) STATE[state_i] * FACT);
#endif
}

/*****************************************************************************/
/*  MRG32k3a code from the paper Good Parameters and Implementations for     */
/*  Combined Multiple Recursive Random Number Generators by Pierre L'Ecuyer  */
/*  with modified scaling of the output suitable for the current use.        */
/*  The original code and the paper may be found at                          */
/*  http://www.iro.umontreal.ca/~lecuyer/papers.html                         */
/*****************************************************************************/

#define norm 2.328306549295728e-10 // only for TestMRG32k3a
#define m1   4294967087.0
#define m2   4294944443.0
#define a12     1403580.0
#define a13n     810728.0
#define a21      527612.0
#define a23n    1370589.0

double  s10, s11, s12, s20, s21, s22;

double MRG32k3a() // scaled to [1, m1 = 2^32 - 209]
{
    long   k;
    double p1, p2;
    /* Component 1 */
    p1 = a12 * s11 - a13n * s10;
    k = p1 / m1;   p1 -= k * m1;   if (p1 < 0.0) p1 += m1;
    s10 = s11;   s11 = s12;   s12 = p1;
    /* Component 2 */
    p2 = a21 * s22 - a23n * s20;
    k  = p2 / m2;  p2 -= k * m2;   if (p2 < 0.0) p2 += m2;
    s20 = s21;   s21 = s22;   s22 = p2;
    /* Combination */
    if (p1 <= p2) return (p1 - p2 + m1); // scaling factor norm removed
    else return (p1 - p2);               // scaling factor norm removed
}

#define hseed 12345.0 // hash function seed

void InitMRG32k3a()
{
    s10 = hseed;
    s11 = hseed;
    s12 = hseed;
    s20 = hseed;
    s21 = hseed;
    s22 = hseed;
}

/*****************************************************************/
/* Initialization of WELLRNG19937a using a linear hash function  */
/* modulo 2^32-5 with coefficients generated by MRG32k3a.        */
/* Petr Savicky 2006                                             */
/*****************************************************************/

#include <math.h>
#define hcount 4
#define hshift 8
#define hmask 0xff
#define hmod 4294967291.0

double aux[R];

void HInitWELLRNG19937a(int nseed, int *vseed)
{
    unsigned int z;
    int i,j,k;
    double small;
    state_i = 0;               // copied from InitWELLRNG19937a
    WELLRNG19937a = case_1;    // copied from InitWELLRNG19937a
    InitMRG32k3a();
    for (k=0; k<R; k++) {
        aux[k] = MRG32k3a();
    }
    for (i=0; i<nseed; i++) {
        z = (unsigned int) vseed[i];
        for (j=0; j<hcount; j++) {
            small = (double)((z&hmask) + 1);
            z >>= hshift;
            for (k=0; k<R; k++) {
                aux[k] += small*MRG32k3a();
            }
        }
    }
    for (k=0; k<R; k++) {
        STATE[k] = (unsigned int)(aux[k] - floor(aux[k]/hmod)*hmod + 1.0);
    }
}

/**********************/
/*  Test functions    */
/**********************/

#include <stdio.h>

void PrintSTATE19937a()
{
    int k;
    printf("printing the current state\n");
    for (k=0; k<R; k++) {
        printf(" %d %u\n",k,STATE[k]);
    }
    printf("end of the current state\n");
}

void TestMRG32k3a()
{
    int i,j;
    double add = 0.0;
    InitMRG32k3a();
    for (i=0; i<10000000; i++) {
        add += MRG32k3a() * norm;
    }
    // The output should be 5001090.95, see the original reference.
    printf("sum test of MRG32k3a: %11.2f\n",add);
}


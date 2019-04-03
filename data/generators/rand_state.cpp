#include <stdio.h>
#include <stdlib.h>
#define e53 9007199254740992.0

struct state {
	double z[32],b;
	unsigned int i,j;
} st;

double em53; 

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

void seed(unsigned int initj)
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

void validate()
{
	int k,l;
	seed(3568225);
	for (k=0; k<1000000; k++) {
		get();
	}
	for (k=0; k<20; k++) {
		for (l=0; l<3; l++) {
			printf(" %24.16e",get());
		}
		printf("\n");
	}
}

void matrix(int s, int m, int n)
{
	int k,l;
	seed((unsigned)s);
	for (k=0; k<m; k++) {
		for (l=0; l<n; l++) {
			printf(" %10.8f",get());
		}
		printf("\n");
	}
}

int main(int argc, char *argv[])
{
	int initj,m,n;
	em53 = 1.0/e53;
    if (argc == 1+1) {
		m = atoi(argv[1]);
		if (m == -1) {
			validate();
			exit(0);
		}
	}
    if (argc != 1+3) {
        printf("parameters: seed m n\n");
        exit(0);
    }
    initj = atoi(argv[1]);
	m = atoi(argv[2]);
	n = atoi(argv[3]);
	matrix(initj,m,n);
}


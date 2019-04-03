  #include <stdio.h>
  
  union str_double
  {
      double value;
      unsigned int word[2];
  };
  
  double copy(double x) { return(x); }
  
  int main(int argc, char *argv[])
  {
      union str_double a,b;
  
      a.word[1] = 0x7ff00001;
      a.word[0] = 0x00000001;
      printf("%08x  %08x\n", a.word[1], a.word[0]); // 7ff00001  00000001
  
      b.value = a.value;
      printf("%08x  %08x\n", b.word[1], b.word[0]); // depends on which PC
  
      b.value = copy(a.value);
      printf("%08x  %08x\n", b.word[1], b.word[0]); // 7ff80001  00000001
  }



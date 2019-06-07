#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ktz.h"

int main (int argc, char **argv) {
  int i = 0;
  struct kts k;
  while (i++<2) {
    printf("\r[%u] reading...", i);
    fflush(stdout);
    readKtz(argv[1], &k, 1);
    printf("\r[%u] freeing...", i);
    fflush(stdout);
    freeKtz(&k);
  }
}

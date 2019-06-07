
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef __sparc
#include <sys/int_types.h>
#else
#include <stdint.h>
#endif

#include "ktz.h"
#include "zlib.h"



// zlib descriptor and buffer
static gzFile gzfh = NULL;
#define BUFFSIZE 131072 // 524288
static char buffer[BUFFSIZE];


// scanner globals
static int READ = 0;           // amount read
static int NEXT = 0;           // current char index in buffer
static unsigned char CHAR = 0; // current character
static char cbuffer[1024];     // temporary buffer for strings. WARNING: size overflow not checked ...
static int64_t inexpected = 0;    // for decoding transition targets




// elementary readers --------------------------------------------------------------------------------

// reads a character in CHAR
// could call gzgetc instead and get rid of buffer, but much faster this way
//    unsigned char readChar () {return CHAR = gzgetc(gzfh);}
unsigned char readChar () {
  int n = NEXT;
  if (n >= READ) {
    READ = gzread(gzfh, buffer, BUFFSIZE);
    if (READ == 0) {
      NEXT = 0;
      CHAR = 0;
    } else if (READ > 0) {
      NEXT = 1;
      CHAR = buffer[0];
    } else {
      int x;
      const char *m = gzerror(gzfh,&x);
      fprintf (stderr,"ERROR (%d, %s)\n",x,m);
      exit(1);
    }
  } else {
    NEXT = 1 + n;
    CHAR = buffer[n];
  }
  return CHAR;
}

// reads k characters in cbuffer
char *readBytes (int k) {
  if (k) {
    int i = 0;
    cbuffer[0] = CHAR;
    for (i=1;i<k;i++) cbuffer[i] = readChar ();
    cbuffer[i] = '\0';
    readChar ();
  } else {
    cbuffer[0] = '\0';
  }
  return cbuffer;
}

// reads n characters as a string
char *readKey (int n) {
  int i = 0;
  for (i=0;i<n;i++) cbuffer[i] = readChar ();
  cbuffer[i] = '\0';
  readChar ();
  return cbuffer;
}

// vle uncompresses nonnegative integer
int32_t rvle (int32_t c) {
  if (c > 127) {
    return 128 * rvle (readChar ()) + (c-128);
  } else {
    readChar ();
    return c;
  }
}

// decoding of nats and ints by vle:
int32_t readNat () {
  if (CHAR == 0) {
    readChar ();
    return 0;
  } else {
    return rvle (CHAR);
  }
}

// vle uncompresses nonnegative long
int64_t rvle64 (int64_t c) {
  if (c > 127) {
    return 128 * rvle64 (readChar ()) + (c-128);
  } else {
    readChar ();
    return c;
  }
}

// decoding of nats and ints by vle:
int64_t readNat64 () {
  if (CHAR == 0) {
    readChar ();
    return 0;
  } else {
    return rvle64 (CHAR);
  }
}
int32_t readInt () {
  int32_t n = readNat ();
  if (n != 0) {
    if (n%2 == 0) {
      return (n/2);
    } else {
      return - ((n+1)/2);
    }
  } else {
    return 0;
  }
}
int64_t readInt64 () {
  int64_t n = readNat64 ();
  if (n != 0) {
    if (n%2 == 0) {
      return (n/2);
    } else {
      return - ((n+1)/2);
    }
  } else {
    return 0;
  }
}

// reads a string in cbuffer
char *readString () {
  return readBytes (readNat ());
}

// reads uncompressed 32 bit integer
int32_t readNAT () {
  int i = 0;
  int32_t N = 0;
  // for (i=0; i<4; i++) {N = 256 * N + (int)CHAR; readChar();}
  for (i=0; i<4; i++) {N = N<<8 | (int32_t)CHAR; readChar();}
  return N;
}

// reads uncompressed 64 bit integer
int64_t readNAT64 () {
  int i = 0;
  int64_t N = 0;
  // for (i=0; i<8; i++) {N = 256 * N + (int64_t)CHAR; readChar();}
  for (i=0; i<8; i++) {N = N<<8 | (int64_t)CHAR; readChar();}
  return N;
}







// ktz reader ----------------------------------------------------------------------------------------

// reads cnt propsets (immediate or ref), for all namespaces in ns
// note: cannot store states on the fly, as some formats store transitions in reverse order
struct propsetPtr *read_props (int cnt, struct namespace *ns) {
  int i;
  struct propsetPtr *ps = (struct propsetPtr *)malloc(cnt*sizeof(struct propsetPtr));
  for (i=0; i<cnt; i++) {
    // read properties for namespace i
    int64_t z = readInt64();
    if (z < 0) {
      // immediate
      struct propset *p = (struct propset *)malloc(sizeof(struct propset));
      int k = (-z) - 1;  // size of property set
      struct prop *pset = (struct prop *)malloc(k*sizeof(struct prop));
      int j;
      for (j=0; j<k; j++) {
	int x = readNat();
	if (x==0) {
	  pset[j].weight = (ns[i].neg==1) ? readInt() : readNat();
	  pset[j].atom = readNat()-1;
	} else {
	  pset[j].weight = (ns[i].inf==1) ? 2 : 1;
	  pset[j].atom = x-1;
	}
      }
      p->size = k;
      p->set = pset;
      ps[i].kind = IMM;
      ps[i].arg.imm = p;
    } else {
      // reference
      ps[i].kind = REF;
      ps[i].arg.ref = z;
    }
  }
  return ps;
}

// generic read_state; no is index of state retreived
struct state read_state (int rev, int64_t no, struct kts *K, int64_t (*readTarget)()) {
  struct state s;
  int i;
  s.no = no;
  // read state properties
  s.sprops = read_props(K->sns,K->sprops);
  // reads transitions and stores them in state s, in reverse order if rev
  s.trcount = readNat();
  struct transition *trs = (struct transition *)malloc(s.trcount*sizeof(struct transition));
  if (rev) {
    // in reverse order
    for (i=s.trcount-1; i >=0; i--) {
      trs[i].tprops = read_props(K->tns,K->tprops);
      trs[i].target = readTarget();
    }
  } else {
    for (i=0; i<s.trcount; i++) {
      trs[i].tprops = read_props(K->tns,K->tprops);
      trs[i].target = readTarget();
    }
  }
  s.trans = trs;
  return s;
}


// state readers

// simple undiff
struct state read_state_diff (int64_t no, struct kts *K) {
  struct state s = read_state (0,no,K,&readNat64);
  // decode targets
  int i, off = 0;
  for (i=0; i<s.trcount; i++) {
    off += s.trans[i].target;
    s.trans[i].target = off;
  }
  return s;
}

// reverse then undiff
struct state read_state_rdiff (int64_t no, struct kts *K) {
  struct state s = read_state (1,no,K,&readNat64);
  // decode targets
  int i, off = 0;
  for (i=0; i<s.trcount; i++) { 
    off += s.trans[i].target;
    s.trans[i].target = off;
  }
  return s;
}

// no is index of state retreived
struct state read_state_raw (int64_t no, struct kts *K) {
  return read_state (0,no,K,&readInt64);
}

struct state read_state_bf (int64_t no, struct kts *K) {
  struct state s = read_state (1,no,K,&readInt64);
  // addi 0 l0 []
  int i, j, g, off = 0;
  for (i=s.trcount-1; i>=0; i--) {
    off -= s.trans[i].target;
    s.trans[i].target = off;
  }
  // decodes targets
  for (i=0; i<s.trcount; i++) {
    g = s.trans[i].target + (inexpected + 1);
    if (!s.trans[i].target) inexpected++;
    s.trans[i].target = g;
  }
  return s;
}

struct state read_state_ooo (int64_t no, struct kts *K) {
  struct state s = read_state_bf (readInt64() + inexpected,K);
  return s;
}

struct state read_state_df (int64_t no, struct kts *K) {
  struct state s = read_state (0,no,K,&readInt64);
  int i, j, g, off;
  // decode targets
  int pos = 0;
  off = no+1;
  for (i=0; i<s.trcount; i++) {
    if (s.trans[i].target <= 0) {
      g = s.trans[i].target + off;
      s.trans[i].target = g;
      off = g;
      pos++;
    } else {
      break;
    }
  }
  // addi (i+1) pos
  off = no+1;
  for (i=pos; i<s.trcount; i++) {
    g = s.trans[i].target;
    s.trans[i].target += off;
    off += g;
  }
  // reverse transitions between 0 and pos-1
  if (pos > 0) {
    j = pos - 1;
    for (i=0; i<j; i++) {
      // swap trans[i] and trans[j]
      struct propsetPtr *props = s.trans[i].tprops;
      g = s.trans[i].target;
      s.trans[i].tprops = s.trans[j].tprops;
      s.trans[i].target = s.trans[j].target;
      s.trans[j].tprops = props;
      s.trans[j].target = g;
      j--;
    }
  }
  return s;
}

// reads a state according to file format
struct state readState (int64_t no, struct kts *K, int xpand) {
  struct state s;
  switch (K->format) {
  case 0: s = read_state_raw (no,K); break;
  case 1: s = read_state_df (no,K); break;
  case 2: s = read_state_bf (no,K); break;
  case 3: s = read_state_ooo (no,K); break;
  case 4: s = read_state_diff (no,K); break;
  case 5: s = read_state_rdiff (no,K); break;
  default: fprintf (stderr, "unknown ktz format, please report it\n"); exit(1);
  }
  if (xpand) {
    // -x flag passed; immediate propsets are stored in namespace.psets and
    // replaced by references in states and transitions
    int i,j;
    // for each state namespace
    for (j=0; j<K->sns; j++) {
      if (s.sprops[j].kind == IMM) {
	K->sprops[j].psets[K->sprops[j].next] = s.sprops[j].arg.imm;
	s.sprops[j].kind = REF;
	s.sprops[j].arg.ref = K->sprops[j].next++;
      }
    }
    // for each event state namespace
    for (j=0; j<K->tns; j++) {
      // for each transition
      for (i=0; i <s.trcount; i++) {
	if (s.trans[i].tprops[j].kind == IMM) {
	  K->tprops[j].psets[K->tprops[j].next] = s.trans[i].tprops[j].arg.imm;
	  s.trans[i].tprops[j].kind = REF;
	  s.trans[i].tprops[j].arg.ref = K->tprops[j].next++;
	}
      }
    }
  }
  return s;
}



// reads ktz file into structure K
void readKtz (char *f, struct kts *K, int xpand) {
  int i;
  // gzopen file
  if (NULL == (gzfh = gzopen(f,"rb"))) {
    fprintf (stderr,"cannot open file %s\n",f);
    exit(1);
  }
  // read header
  int j,k,l = 0;
  // -- read and check magic
  char *s;
  s = readKey(7); K->magic = (char *)malloc(1+strlen(s)); strcpy(K->magic,s);
  if (strcmp(s,"KTS0011")) {
    fprintf (stderr,"cannot read ktz file, unknown file format (%s), sorry\n",s);
    exit(1);
  }
  // -- sizes (not vle compressed)
  K->nodes = readNAT64();
  K->edges = readNAT64();
  K->sns = k = readNAT();
  K->sprops = (struct namespace *)malloc(k*sizeof(struct namespace));
  for (i=0; i<k; i++) {
    K->sprops[i].count = readNAT64();
    if (xpand) {
      K->sprops[i].next = 0;
      K->sprops[i].psets = (struct propset **)malloc(K->sprops[i].count*sizeof(struct propset *));
    } else {
      // implicit but does not harm ...
      K->sprops[i].psets = NULL;
    }
  }
  K->tns = k = readNAT();
  K->tprops = (struct namespace *)malloc(k*sizeof(struct namespace));
  for (i=0; i<k; i++) {
    K->tprops[i].count = readNAT64();
    if (xpand) {
      K->tprops[i].next = 0;
      K->tprops[i].psets = (struct propset **)malloc(K->tprops[i].count*sizeof(struct propset *));
    } else {
      // implicit but does not harm ...
      K->tprops[i].psets = NULL;
    }
  }
  // -- comment
  s = readString(); K->label = (char *)malloc(1+strlen(s)); strcpy(K->label,s);
  // -- state encoding option
  K->format = readNat();
  // -- name of kts
  s = readString(); K->name = (char *)malloc(1+strlen(s)); strcpy(K->name,s);
  // -- state namespaces
  k = readNat();
  for (i=0; i<k; i++) {
    // loading ith namespace
    s = readString(); K->sprops[i].name = (char *)malloc(1+strlen(s)); strcpy (K->sprops[i].name,s);
    j = (readNat()) % 4;  // flags
    K->sprops[i].neg = (j/2)>0?1:0;
    K->sprops[i].inf = (j%2)>0?1:0;
    j = readNat();
    K->sprops[i].atoms = j;
    K->sprops[i].names = (char **)malloc(j*sizeof(char *));
    for (l=0; l<j; l++) {
      s = readString();
      K->sprops[i].names[l] = (char *)malloc(1+strlen(s));
      strcpy(K->sprops[i].names[l],s);
    }
  }
  // -- trans namespaces
  k = readNat(); // number of namespaces [redundant, = K->tns ...]
  for (i=0; i<k; i++) {
    // loading ith namespace
    s = readString(); K->tprops[i].name = (char *)malloc(1+strlen(s)); strcpy (K->tprops[i].name,s);
    j = (readNat()) % 4;  // flags
    K->tprops[i].neg = (j/2)>0?1:0;
    K->tprops[i].inf = (j%2)>0?1:0;
    j = readNat();
    K->tprops[i].atoms = j;
    K->tprops[i].names = (char **)malloc(j*sizeof(char *));
    for (l=0; l<j; l++) {
      s = readString();
      K->tprops[i].names[l] = (char *)malloc(1+strlen(s));
      strcpy(K->tprops[i].names[l],s);
    }
  }
  // read states
  K->g = (struct state *)malloc(K->nodes*sizeof(struct state));
  for (i=0; i<K->nodes; i++)
    K->g[i] = readState (i,K,xpand);
  // gzclose file
  gzclose(gzfh);
}




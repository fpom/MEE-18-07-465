
// types

struct prop {
  int atom;                      // atomic prop index
  int weight;                    // prop weight
};

struct propset {
  int size;                      // size of propset
  struct prop *set;              // array of size props
};

struct namespace {
  char *name;                    // name of namespace
  int neg;                       // != 0 if values may be negative
  int inf;                       // != 0 if values may be infinite
  int64_t count;                 // number of propsets of that namespace found in ktz (or bound)
  int atoms;                     // number of atomic properties in namespace
  char **names;                  // their names as an array of size strings
  // next two for implementing -x option
  int64_t next;                  // first available slot in psets
  struct propset **psets;        // array of propsets pointers
};

struct propsetPtr {              // propsets as read or written in ktz
  enum {IMM,REF} kind;           // propset or reference
  union {
    int64_t ref;                 // reference
    struct propset *imm;         // propset ptr
  } arg;
};

struct transition {
  struct propsetPtr *tprops;     // array of propsetPtr (one per event namespace)
  int64_t target;                // target state number
};

struct state {
  int64_t no;                    // state number
  struct propsetPtr *sprops;     // array of propsetsPtr (one per state namespace)
  int trcount;                   // number of transitions
  struct transition *trans;      // array of trcount transitions
};

struct kts {
  char *magic;                   // magic
  char *label;                   // comment
  char *name;                    // name of kts
  int64_t nodes;                 // number of nodes in kts
  int64_t edges;                 // number of edges in kts
  int format;                    // format (in 0..5, encoding bf, df, etc)
  int32_t sns;                   // number of state property namespaces
  struct namespace *sprops;      // array of sns state property namespaces
  int32_t tns;                   // number of event property namespaces
  struct namespace *tprops;      // array of tns event property namespaces
  struct state *g;               // array of states
};



// functions

void readKtz(char *, struct kts *, int);
// void writeKtz(char *, struct kts *, int, int, int);
// void printKts(struct kts, int);
void freeKtz (struct kts *);

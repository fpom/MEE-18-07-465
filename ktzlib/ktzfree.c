
#include <stdio.h>
#include <stdlib.h>
#ifdef __sparc
#include <sys/int_types.h>
#else
#include <stdint.h>
#endif

#include "ktz.h"

// #define DEBUG
#ifdef DEBUG
#define PRINTF(...) {printf (__VA_ARGS__);}
#else
#define PRINTF(...)
#endif

void freePropSet (struct propset *p) {
  int i;
  PRINTF ("freeing propset\n");
  free (p->set);
  free (p);
}

void freeNamespace (struct namespace *ns) {
  int32_t i;
  PRINTF ("freeing namespace %s\n", ns->name);
  PRINTF ("atoms=%d, count=%d, next=%d\n", ns->atoms, ns->count, ns->next);
  free (ns->name);
  // freeing names
  for (i=0; i<ns->atoms; i++) {
    PRINTF ("name=%s\n",ns->names[i]);
    free (ns->names[i]);
  }
  free (ns->names);
  // freeing propsets
  for (i=0; i<ns->next; i++) {
    freePropSet (ns->psets[i]);
  }
  free (ns->psets);
}

void freePropsetPtr (struct propsetPtr *p) {
  PRINTF ("freeing propsetPtr\n");
  if (p->kind == IMM) {
    freePropSet (p->arg.imm);
  }
}

void freeTransition (struct transition *t, int32_t tns) {
  int32_t i;
  for (i=0; i<tns; i++) freePropsetPtr(&t->tprops[i]);
  free (t->tprops);
}

void freeState (struct state *s, int32_t sns, int32_t tns) {
  int32_t i;
  PRINTF ("\nfreeing state\n");
  for (i=0; i<sns; i++) freePropsetPtr(&s->sprops[i]);
  free (s->sprops);
  PRINTF ("freeing transitions\n");
  for (i=0; i<s->trcount; i++) freeTransition(&s->trans[i], tns);
  free(s->trans);
}

void freeKtz (struct kts *k) {
  int i;
  PRINTF ("freeing kts\n");
  free (k->magic);
  PRINTF ("magic freed\n");
  free (k->label);
  PRINTF ("label freed\n");
  free (k->name);
  PRINTF ("name freed\n");
  for (i=0; i<k->sns; i++) freeNamespace (&k->sprops[i]);
  free (k->sprops);
  PRINTF ("s-namespaces freed\n");
  for (i=0; i<k->tns; i++) freeNamespace (&k->tprops[i]);
  free (k->tprops);
  PRINTF ("e-namespaces freed\n");
  for (i=0; i<k->nodes; i++) freeState (&k->g[i],k->sns,k->tns);
  free (k->g);
  PRINTF ("\nstates freed\n");
  // free (k);  not freed; assumed locally allocated in caller
  PRINTF ("kts freed\n");
}

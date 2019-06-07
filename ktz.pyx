import sys, csv, os.path, os, ast, time, datetime, operator, collections, bz2
import pandas

from tables import PathPool, TableReader, TableWriter

cimport numpy as np
import numpy as np

##############################################################################
##############################################################################
####                                                                      ####
#### ktz.h                                                                ####
####                                                                      ####
##############################################################################
##############################################################################

cdef extern from "ktz.h" :
    cdef struct prop :
        int atom
        int weight
    cdef struct propset :
        int size
        prop *set
    cdef struct namespace :
        char *name
        int neg
        int inf
        int count
        int atoms
        char **names
        int next
        propset **psets
    cdef enum _kind :
        IMM, REF
    cdef union _arg :
        int ref
        propset *imm
    cdef struct propsetPtr :
        _kind kind
        _arg arg
    cdef struct transition :
        propsetPtr *tprops
        int target
    cdef struct state :
        int no
        propsetPtr *sprops
        int trcount
        transition *trans
    cdef struct kts :
        char *magic
        char *label
        char *name
        int nodes
        int edges
        int format
        int sns
        namespace *sprops
        int tns
        namespace *tprops
        state *g
    cdef void readKtz(char *, kts *, int)
    cdef void freeKtz(kts *)

##############################################################################
##############################################################################
####                                                                      ####
#### auxiliary stuff                                                      ####
####                                                                      ####
##############################################################################
##############################################################################

cdef void log (str message, int nl=0) :
    sys.stderr.write("\r" + message.ljust(70))
    if nl :
        sys.stderr.write(b"\n")
    sys.stderr.flush()

cpdef enum status :
    INIT, SCC, SCCINIT, BASIN, DEAD

cpdef bytes status_str (status s) :
    cdef dict _s = {status.INIT : b"INIT",
                    status.SCC : b"SCC",
                    status.SCCINIT : b"SCCINIT",
                    status.BASIN : b"BASIN",
                    status.DEAD : b"DEAD"}
    return _s[s]

cpdef status status_val (bytes s) :
    cdef dict _s = {b"INIT" : status.INIT,
                    b"SCC" : status.SCC,
                    b"SCCINIT" : status.SCCINIT,
                    b"BASIN" : status.BASIN,
                    b"DEAD" : status.DEAD}
    return _s[s.upper()]

cdef class IntQueue :
    cdef object q
    cdef set s
    def __cinit__ (self) :
        self.q = collections.deque()
        self.s = set()
    cpdef void put (self, int n) :
        if n not in self.s :
            self.q.append(n)
            self.s.add(n)
    cpdef int get (self) :
        cdef int n = self.q.popleft()
        self.s.remove(n)
        return n
    cpdef int size (self) :
        return len(self.s)
    def __bool__ (self) :
        return bool(self.s)
    def __len__ (self) :
        return len(self.s)

cdef bytes joincell (object s) :
    return b",".join(bytes(x) for x in s)

cdef list splitcell (bytes b, convert=bytes) :
    cdef bytes x
    if b :
        return [convert(x) for x in b.split(b",")]
    else :
        return []

##############################################################################
##############################################################################
####                                                                      ####
#### graph algorithms                                                     ####
####                                                                      ####
##############################################################################
##############################################################################

cdef class _context :
    cdef dict g
    cdef list s
    cdef set  ss
    cdef dict index
    cdef dict lowlink
    cdef list t
    cdef list ret
    def __init__ (self, dict g) :
        self.g = g
        self.s = []
        self.ss = set()
        self.index = {}
        self.lowlink = {}
        self.t = []
        self.ret = []

cdef set empty = set()

cdef void _tarjan_head (_context ctx, int v) :
    cdef set it
    ctx.index[v] = len(ctx.index)
    ctx.lowlink[v] = ctx.index[v]
    ctx.s.append(v)
    ctx.ss.add(v)
    if v in ctx.g :
        ctx.t.append((ctx.g[v], 0, v, -1))
    else :
        ctx.t.append((empty, 0, v, -1))

cdef void _tarjan_body (_context ctx, set it, int v):
    cdef int w
    cdef list scc
    for w in it :
        if w not in ctx.index:
            ctx.t.append((it, 1, v, w))
            _tarjan_head(ctx, w)
            return
        if w in ctx.ss :
            ctx.lowlink[v] = min(ctx.lowlink[v], ctx.index[w])
    if ctx.lowlink[v] == ctx.index[v]:
        scc = []
        w = -1
        while v != w:
            w = ctx.s.pop()
            scc.append(w)
            ctx.ss.remove(w)
        ctx.ret.append(scc)

cpdef list tarjan (dict g):
    cdef set it
    cdef int inside, v, w
    cdef _context ctx = _context(g)
    for v in g :
        if v not in ctx.index :
            _tarjan_head(ctx, v)
        while ctx.t :
            it, inside, v, w = ctx.t.pop()
            if inside:
                ctx.lowlink[v] = min(ctx.lowlink[w], ctx.lowlink[v])
            _tarjan_body(ctx, it, v)
    return ctx.ret

cpdef tuple merge_classes (object states, object trans, list classes, bytes col) :
    cdef int i, n, s, d
    cdef set cls, rules
    cdef dict cnum = {}
    cdef dict succ = {}
    cdef dict pred = {}
    cdef set onoff
    cdef dict on = {}
    cdef dict off = {}
    cdef np.ndarray[np.int_t, ndim=1] ints
    cdef np.ndarray[bytes, ndim=1] strings
    cdef bytes b
    cdef list states_, trans_
    classes.sort(key=min)
    for n, cls in enumerate(classes) :
        for s in cls :
            cnum[s] = n
    for s, cls in enumerate(classes) :
        ints = trans[trans.src.isin(cls)].dst.values
        succ[s] = set()
        for i in range(ints.shape[0]) :
            n = cnum[ints[i]]
            if n != s :
                succ[s].add(n)
    for s, cls in succ.iteritems() :
        for d in cls :
            if d not in pred :
                pred[d] = set()
            pred[d].add(s)
    row = states.itertuples().next()
    onoff = (set(row.on.split(",")) | set(row.off.split(","))) - set([""])
    for n, cls in enumerate(classes) :
        on[n] = onoff.copy()
        strings = states[states.node.isin(cls) & ~states.on.isnull()].on.values
        for i in range(strings.shape[0]) :
            on[n].intersection_update(strings[i].split(b","))
        off[n] = onoff.copy()
        strings = states[states.node.isin(cls) & ~states.off.isnull()].off.values
        for i in range(strings.shape[0]) :
            off[n].intersection_update(strings[i].split(b","))
    states_ = []
    for n, cls in enumerate(classes) :
        sub = states[states.node.isin(cls)]
        states_.append([n,
                        ",".join(map(str, succ.get(n, []))),
                        ",".join(map(str, pred.get(n, []))),
                        ",".join(on.get(n, [])),
                        ",".join(off.get(n, [])),
                        sub.input.any(),
                        sub.output.any(),
                        iter(sub[col]).next()])
    trans_ = []
    for s in succ :
        for d in succ[s] :
            rules = set()
            strings = trans[trans.src.isin(classes[s])
                            & trans.dst.isin(classes[d])].rules.values
            for i in range(strings.shape[0]) :
                rules.update(strings[i].split(b","))
            trans_.append([s, d, b",".join(rules)])
    return (pandas.DataFrame(columns=["node", "succ", "pred", "on", "off", "init",
                                      "dead", col],
                             data=states_),
            pandas.DataFrame(columns=["src", "dst", "rules"],
                             data=trans_))

##############################################################################
##############################################################################
####                                                                      ####
#### load ktz, build components, save files                               ####
####                                                                      ####
##############################################################################
##############################################################################

cdef list get_props (propsetPtr pp, namespace ns) :
    cdef propset *p
    cdef list props = []
    cdef int i, w, a
    cdef bytes n
    if pp.kind == IMM :
        p = pp.arg.imm
    else :
        p = ns.psets[pp.arg.ref]
    for i in range(p.size) :
        w = p.set[i].weight
        a = p.set[i].atom
        n = ns.names[a]
        if ns.inf and w == 1 :
            w = sys.maxint
        elif ns.inf :
            w -= 1
        props.append((n, w))
    return props

cpdef tuple load_ktz (char *path, int progress=0, int quick=0) :
    cdef kts k
    cdef state *s
    cdef transition *t
    cdef int snum, tnum, n, m, pval, i, j, step, size, sccnum
    cdef int hasscc = 0
    cdef list props
    cdef bytes pname
    cdef dict succ = {}
    cdef dict pred = {}
    cdef dict scc = {}
    cdef set init = set()
    cdef set dead = set()
    cdef set drop = set()
    cdef set on, off, c
    cdef list sccfound = []
    cdef namespace *ns
    if progress :
        log("reading...")
    readKtz(path, &k, 1)
    ns = k.sprops
    for i in range(ns.atoms) :
        if ns.names[i] == b"scc" :
            hasscc = 1
            break
    if quick :
        hasscc = False
    if progress :
        size = k.nodes
        step = max(1, size / 100)
    # prepare CSV
    if not quick :
        pp = PathPool(path)
        csv_n = TableWriter(pp.nodes(),
                            ["node", "succ", "pred", "on", "off", "init", "dead"])
        csv_e = TableWriter(pp.edges(), ["src", "dst", "rules"])
    # preprocess each state
    for i in range(k.nodes) :
        # get state
        if progress and i % step == 0 :
            log("loaded %u%% (transient %u%%)"
                % (100 * i / size, 100 * len(drop) / size))
        s = &(k.g[i])
        snum = s.no
        if i == 0 :
            init.add(snum)
        # get places
        if not quick :
            on = set()
            off = set()
            sccnum = 0
            for j in range(k.sns) :
                props = get_props(s.sprops[j], k.sprops[j])
                for pname, pval in props :
                    if pname == b"scc" :
                        sccnum = pval
                    elif pname[-1] == b"+" :
                        on.add(pname[:-1])
                    elif pname[-1] == b"-" :
                        off.add(pname[:-1])
            if hasscc :
                if sccnum not in scc :
                    scc[sccnum] = set()
                scc[sccnum].add(snum)
        # get transitions
        if s.trcount == 0 :
            dead.add(snum)
        else :
            succ[snum] = set()
        for j in range(s.trcount) :
            # get trans
            t = &(s.trans[j])
            tnum = t.target
            if tnum == snum :
                continue
            succ[snum].add(tnum)
            if tnum not in pred :
                pred[tnum] = set()
            pred[tnum].add(snum)
            props = get_props(t.tprops[0], k.tprops[0])
            pname = props[0][0].rsplit(":", 1)[-1]
            if not quick :
                csv_e.writerow([snum, tnum, pname])
            if pname[0] in "cC" :
                drop.add(snum)
        if not quick :
            csv_n.writerow([snum,
                            joincell(succ.get(snum, [])),
                            joincell(pred.get(snum, [])),
                            ",".join(on), ",".join(off),
                            snum in init, snum in dead])
    if not quick :
        csv_e.close()
        csv_n.close()
    #freeKtz(&k)
    if hasscc :
        sccnum = 0
        for c in scc.itervalues() :
            if len(c) > 1 :
                sccfound.append(c)
                sccnum += len(c)
    if progress :
        if hasscc :
            log("loaded %u states (transient %u, %s SCC = %u)"
                % (size, len(drop), len(sccfound), sccnum), 1)
        else :
            log("loaded %u states (transient %u)" % (size, len(drop)), 1)
    if hasscc :
        return size, succ, pred, init, dead, sccfound, drop
    elif quick :
        return size - len(drop), succ, pred, init, dead, drop
    else :
        return size, succ, pred, init, dead, None, drop

cdef void _add_edges_from (object g, dict s) :
    cdef int i, j
    g.add_edges_from((i, j) for i in s for j in s[i])

cpdef list build_scc (dict succ, int progress=0) :
    cdef list l
    cdef list scc = []
    if progress :
        log("here comes Tarjan...")
    for l in tarjan(succ) :
        if len(l) > 1 :
            scc.append(set(l))
    scc.reverse()
    if progress :
        log("Tarjan found %u SCC" % len(scc), 1)
    return scc

cpdef tuple build_basins (int size, dict succ, dict pred,
                          set init, set dead, list scc,
                          int progress=0) :
    cdef int i, s, p, step
    cdef int nbr = 0
    cdef status k
    cdef set c
    cdef frozenset f
    cdef IntQueue todo = IntQueue()
    cdef dict done = {}
    cdef dict comp = {}
    cdef dict ret = {}
    if progress :
        log("building initial components...")
        step = max(1, size / 100)
    # create first components: scc
    for c in scc :
        if c & init :
            k = status.SCCINIT
        else :
            k = status.SCC
        comp[frozenset([nbr])] = (k, nbr)
        for s in c :
            done[s] = nbr
        nbr += 1
    # create first components: init if needed
    for s in init :
        if s not in done :
            comp[frozenset([nbr])] = (status.INIT, nbr)
            done[s] = nbr
            nbr += 1
    # create first components: deadlocks
    for s in dead :
        comp[frozenset([nbr])] = (status.DEAD, nbr)
        done[s] = nbr
        nbr += 1
    # prepare todo = pred(done) - done
    for s in done :
        if s in pred :
            for p in pred[s] :
                if p not in done :
                    todo.put(p)
    # compute for each unclassified node
    i = 0
    while todo :
        if progress :
            if i % step == 0 :
                log("computing basins %u%%" % (100 * i / size))
            i += 1
        s = todo.get()
        c = set()
        for p in succ[s] :
            if p in done :
                c.add(done[p])
        f = frozenset(c)
        if f not in comp :
            comp[f] = (status.BASIN, nbr)
            nbr += 1
        done[s] = comp[f][1]
        if s in pred :
            for p in pred[s] :
                if p not in done :
                    todo.put(p)
    if progress :
        log("computed %u components" % len(comp), 1)
    for k, n in comp.itervalues() :
        ret[n] = k
    return ret, done

cpdef save_comp (char *path, dict comp, int progress=0) :
    cdef int size, step
    if progress :
        log("loading table to update...")
    table = TableReader.dataframe(path)
    if progress :
        size = len(table)
        step = max(1, size / 100)
    with TableWriter(path, list(table.columns) + ["component"]) as out :
        for row in table.itertuples() :
            if progress and row.Index % step == 0 :
                log("saving components for nodes %u%%" % (100 * row.Index / size))
            out.writerow(row[1:] + (comp[row.node],))
    if progress :
        log("saved components for %u nodes" % size, 1)

cpdef void merge_nodes (char *path, dict succ, pred, dict comp, dict kind,
                        int progress=0) :
    cdef dict mbr = {}
    cdef tuple e
    cdef int s, t, n
    cdef status k
    cdef set _succ, _pred
    if progress :
        log("inverting...")
    for n in kind :
        mbr[n] = set()
    for s, n in comp.iteritems() :
        mbr[n].add(s)
    if progress :
        log("merging nodes...")
    out = TableWriter(path, ["node", "succ", "pred", "kind", "size"])
    for n, k in kind.iteritems() :
        _succ = set()
        _pred = set()
        for s in mbr[n] :
            if s in succ :
                for t in succ[s] :
                    _succ.add(comp[t])
            if s in pred :
                for t in pred[s] :
                    _pred.add(comp[t])
        _succ.discard(n)
        _pred.discard(n)
        out.writerow([n, joincell(_succ), joincell(_pred), status_str(k), len(mbr[n])])
    if progress :
        log("merged to %u nodes" % len(kind), 1)

cpdef merge_states (char *statepath, char *compopath, dict comp, int progress=0) :
    cdef int size, step, i, n
    cdef dict on = {}
    cdef dict off = {}
    if progress :
        size = len(comp)
        step = max(1, size / 100)
        i = 0
    with TableReader(statepath) as src :
        for row in src :
            if progress :
                if i % step == 0 :
                    log("merging states %u%%" % (100 * i / size))
                i += 1
            n = comp[int(row.node)]
            if n not in on :
                on[n] = set(splitcell(row.on))
                off[n] = set(splitcell(row.off))
            else :
                on[n].intersection_update(splitcell(row.on))
                off[n].intersection_update(splitcell(row.off))
    if progress :
        i = 0
    with TableReader(compopath, True) as src :
        with TableWriter(compopath, src.cols + ("on", "off")) as dst :
            for row in src :
                if progress :
                    if i % step == 0 :
                        log("saving %u%%" % (100 * i / size))
                    i += 1
                n = int(row.node)
                dst.writerow(row + (b",".join(on[n]), b",".join(off[n])))
    if progress :
        log("saved merged states for %u nodes" % size, 1)

cpdef merge_edges (char *src, char *dst, dict comp, int size, int progress=0) :
    cdef dict edges = {}
    cdef int s, d, step, i, sz
    cdef set r
    if progress :
        step = max(1, size / 100)
        i = 0
    with TableReader(src) as table :
        for row in table :
            if progress :
                if i % step == 0 :
                    log("merging edges %u%%" % (100 * i / size))
                i += 1
            s = comp[int(row.src)]
            d = comp[int(row.dst)]
            if s != d :
                if (s, d) not in edges :
                    edges[s, d] = set()
                edges[s, d].update(row.rules.split(b","))
    if progress :
        sz = len(edges)
        step = max(1, sz / 100)
        i = 0
    with TableWriter(dst, ["src", "dst", "rules"]) as out :
        for (s, d), r in edges.iteritems() :
            if progress :
                if i % step == 0 :
                    log("saving %u%%" % (100 * i / sz))
                i += 1
            out.writerow([s, d, b",".join(r)])
    if progress :
        log("merged %s transitions into %s edges" % (size, sz), 1)

cdef set closure (int root, dict succ, set drop) :
    cdef set c = set()
    cdef set done = set()
    cdef set todo = set([root])
    cdef int s, n
    while todo :
        s = todo.pop()
        done.add(s)
        if s in succ :
            for n in succ[s] :
                if n in drop :
                    if n not in done :
                        todo.add(n)
                else :
                    c.add(n)
    return c

cpdef tuple drop_transient (dict succ, dict pred, set init, set drop, int progress=0) :
    cdef dict _succ = {}
    cdef dict _pred = {}
    cdef dict _repl = {}
    cdef set _init = init.copy()
    cdef int s, n
    cdef set c
    cdef int size, step, i
    _init.difference_update(drop)
    if progress :
        size = len(succ)
        step = max(1, size / 100)
        i = 0
    for s, c in succ.iteritems() :
        if progress :
            if i % step == 0 :
                log("building non-transient successors %u%%" % (100 * i / size))
            i += 1
        if s not in drop :
            _succ[s] = c.copy()
    if progress :
        size = len(drop)
        step = max(1, size / 100)
        i = 0
    for s in drop :
        if progress :
            if i % step == 0 :
                log("removing transient states %u%%" % (100 * i / size))
            i += 1
        c = closure(s, succ, drop)
        if s in init :
            _init.update(c)
        if s in pred :
            _repl[s] = c
            for n in pred[s] :
                if n not in drop :
                    _succ[n].discard(s)
                    _succ[n].update(c)
    if progress :
        size = len(_succ)
        step = max(1, size / 100)
        i = 0
    for s in _succ :
        if progress :
            if i % step == 0 :
                log("building predecessors %u%%" % (100 * i / size))
            i += 1
        for n in _succ[s] :
            if n not in _pred :
                _pred[n] = set()
            _pred[n].add(s)
    if progress :
        log("removed %s transient-states" % len(drop), 1)
    return _init, _succ, _pred, _repl

cpdef void save_transient (char *nodes_src, char *nodes_dst,
                           char *edges_src, char *edges_dst,
                           int size, set drop, set init, set dead,
                           dict succ, dict pred, dict repl, int progress=0) :
    cdef list cols
    cdef int n, s, t, edges, step, i
    if progress :
        edges = 0
        step = max(1, size / 100)
        i = 0
    cols = ["node", "succ", "pred", "on", "off", "init", "dead"]
    with TableReader(nodes_src) as src :
        with TableWriter(nodes_dst, cols) as dst :
            for row in src :
                n = int(row.node)
                if progress :
                    if i % step == 0 :
                        log("filtering transient states %u%%" % (100 * i / size))
                    i += 1
                    if n in succ :
                        edges += len(succ[n])
                if n not in drop :
                    dst.writerow([row.node,
                                  joincell(succ.get(n, [])),
                                  joincell(pred.get(n, [])),
                                  row.on, row.off, n in init, n in dead])
    cols = ["src", "dst", "rules"]
    if progress :
        step = max(1, edges / 100)
        i = 0
    with TableReader(edges_src) as src :
        with TableWriter(edges_dst, cols) as dst :
            for row in src :
                if progress :
                    if i % step == 0 :
                        log("filtering transient edges %u%%" % (100 * i / edges))
                    i += 1
                n = int(row.src)
                s = int(row.dst)
                if n not in drop :
                    if s not in drop :
                        dst.writerow(row)
                    else :
                        for t in repl[s] :
                            dst.writerow([n, t, row.rules])
    if progress :
        log("filtered %s nodes and %s transitions" % (size, edges), 1)

def needs_update (targets, sources) :
    mtime = 0.0
    for path in targets :
        if not os.path.isfile(path) :
            return True
        mtime = max(mtime, os.stat(path).st_mtime)
    for path in sources :
        if mtime < os.stat(path).st_mtime :
            return True
    return False

cpdef void build (char *path, int progress=0, int force=0) :
    cdef int e
    cdef set s
    pp = PathPool(path)
    dead = drop = init = succ = pred = size = None
    sources = [pp.ktz]
    for reduced in [False, True] :
        targets = [pp.nodes(reduced=reduced),
                   pp.edges(reduced=reduced),
                   pp.nodes(reduced=reduced, components=True),
                   pp.edges(reduced=reduced, components=True)]
        if force or needs_update(targets, sources) :
            if progress :
                if reduced :
                    log("# removing transient states", 1)
                else :
                    log("# building full graphs", 1)
            if not reduced :
                size, succ, pred, init, dead, scc, drop = load_ktz(pp.ktz, progress)
            else :
                if dead is None :
                    size, succ, pred, init, dead, drop = load_ktz(pp.ktz, progress,
                                                                  quick=True)
                init, succ, pred, repl = drop_transient(succ, pred, init, drop,
                                                        progress)
                save_transient(pp.nodes(reduced=False), pp.nodes(reduced=True),
                               pp.edges(reduced=False), pp.edges(reduced=True),
                               size + len(drop), drop, init, dead, succ, pred, repl,
                               progress)
                scc = None
            if scc is None :
                scc = build_scc(succ, progress)
            kind, comp = build_basins(size, succ, pred, init, dead, scc, progress)
            save_comp(pp.nodes(reduced=reduced), comp, progress)
            merge_nodes(pp.nodes(reduced=reduced, components=True),
                        succ, pred, comp, kind, progress)
            merge_states(pp.nodes(reduced=reduced),
                         pp.nodes(reduced=reduced, components=True),
                         comp, progress)
            e = 0
            for s in succ.itervalues() :
                e += len(s)
            merge_edges(pp.edges(reduced=reduced),
                        pp.edges(reduced=reduced, components=True), comp, e, progress)
        sources = targets

import sys, os, os.path, inspect, logging
import hlr, ktz, rr2pn, analyse, tables

logging.captureWarnings(True)

def mtime (path) :
    return os.stat(path).st_mtime

def _doc (cls, indent="") :
    for name, method in sorted((n, m) for n, m in
                               inspect.getmembers(cls, inspect.ismethod)
                               if n.startswith("_make_")
                               and m.__doc__) :
        if "=>" in method.__doc__ :
            doc, ret = (x.strip() for x in method.__doc__.rsplit("=>", 1))
        else :
            doc = method.__doc__
            ret = ""
        if indent :
            dot = "."
        else :
            dot = ""
        yield "    " + indent + dot + name[6:] + " => " + doc
        if ret :
            for line in globals()[ret]._doc(indent + "  ") :
                yield line

class Lazy (object) :
    def __init__ (self, obj=None) :
        self._obj = obj
        self._args = {}
    def __call__ (self, **args) :
        self._args = args
        return self
    def __getattr__ (self, name) :
        try :
            if hasattr(self, "_make_" + name) :
                obj = getattr(self, "_make_" + name)(**self._args)
                if obj is not self and obj is not None :
                    setattr(self, name, obj)
                return obj
            elif hasattr(self._obj, name) :
                return getattr(self._obj, name)
            else :
                raise AttributeError("object '%s' has no attribute '%s'"
                                     % (self.__class__.__name__, name))
        finally :
            self._args = {}
    @classmethod
    def doc (cls) :
        return "\n".join(cls._doc())
    @classmethod
    def _doc (cls, indent="") :
        return _doc(cls, indent)
    def _make_help (self) :
        print "Available methods:"
        print self.doc()
        return self

class Model (Lazy) :
    def __init__ (self, path) :
        self.path = path
        Lazy.__init__(self)
    def _make_spec (self) :
        "load SPEC and compute the rules"
        print "Loading %r" % self.path
        rr2pn.Rule._count = 0
        spec = rr2pn.Spec.parse(self.path)
        for rule in spec.constraints + spec.rules :
            print " ", rule
        return spec
    def _make_ring (self) :
        "build the ecosystemic graph => Graph"
        return Ring(self.spec, self.spec.ring(self.spec.init()))
    def _make_size (self) :
        "compute the number of states/transitions in the full state space"
        try :
            print "  %s states, %s edges" % self.spec.size()
        except OSError :
            print "  states count not available, you must install Tina and ITS-tools"
        return self
    def _make_net (self) :
        "compute the Petri net => PetriNet"
        self.spec
        print "Building Petri net"
        net = self.spec.net()
        print "  %s places, %s transitions" % (len(net.place()), len(net.transition()))
        self.size
        return PetriNet(net)
    def _states_or_reduced (self, reduced) :
        print "Building ktz"
        pp = tables.PathPool(self.spec.path)
        if ktz.needs_update([pp.net, pp.ktz], [pp.rr]) :
            print "  creating %r..." % pp.net
            self.spec.tina()
            print "  started tina..."
            self.spec.ktz(pp.ktz, True)
        return Ktz(self.spec, reduced)
    def _make_full (self) :
        "full state space => Ktz"
        return self._states_or_reduced(False)
    def _make_tred (self) :
        "transient-reduced state space => Ktz"
        return self._states_or_reduced(True)

class Ktz (Lazy) :
    def __init__ (self, obj, reduced) :
        Lazy.__init__(self, obj)
        self.reduced = reduced
    def _make_build (self) :
        "ensure that data related to a model is built and up-to-date"
        ktz.build(self._obj.path, progress=True, force=False)
        return self
    def _make_rebuild (self) :
        "force rebuild data related to a model"
        ktz.build(self._obj.path, progress=True, force=True)
        return self
    def _make_palette (self) :
        "colors palette for components => Graph"
        return Graph(analyse.Graph.palette())
    def _make_cg (self, engine="neato") :
        "components graph => Graph"
        self._make_build()
        return Graph(analyse.MergedGraph.load(self._obj.path, reduced=True,
                                              engine=engine))
    def _make_sg (self, engine="neato") :
        "graph of the full statespace => Graph"
        self._make_build()
        return Graph(analyse.StateSpace.load(self._obj.path, reduced=True,
                                             engine=engine))
    def _make_compo (self) :
        "set of components to access each individually => ComponentSet"
        self._make_build()
        return ComponentSet(analyse.ComponentSet(self._obj.path, self.reduced))

class ComponentSet (Lazy) :
    def __init__ (self, *l, **k) :
        Lazy.__init__(self, *l, **k)
        self._items = {}
    @classmethod
    def _doc (cls, indent="") :
        for line in _doc(cls, indent) :
            yield line
        yield "    " + indent + "[NUM] => extract component #NUM"
        for line in Component._doc(indent + "    ") :
            yield line
    def _make_ls (self) :
        "list components and their sizes"
        print "Listing components:"
        if not hasattr(self, "_ls") :
            self._ls = self._obj.ls()
        for i, l in enumerate(self._ls) :
            print "  #%s => %s nodes" % (i, l)
        return self
    def __getitem__ (self, idx) :
        if idx in self._items :
            return self._items[idx]
        self._items[idx] = comp = Component(self._obj[idx], idx)
        return comp

class Component (Lazy) :
    @classmethod
    def _doc (cls, indent="") :
        for line in _doc(cls, indent) :
            yield line
        yield "    " + indent + "[KEY]=FUN => add attr KEY to nodes as computed by FUN"
        yield "    " + indent + "             calling FUN(**node_attributes)"
        yield "    " + indent + "[KEY] => merge component wrt KEY"
        yield "    " + indent + "[KEY,True] => collapse component wrt KEY"
    def __init__ (self, obj, num, col=None) :
        Lazy.__init__(self, obj)
        self.num = num
        self.col = col
        self._items = {}
    def _make_save (self) :
        "save component to CSV+BZ2"
        for path in self._obj.save() :
            print "  >", path
        return self
    def _make_info (self) :
        "information about the component"
        print "Component %r (%s):" % (self._obj.name, self._obj.kind)
        print "  #%s in components set" % self.num
        print "  %s nodes" % len(self._obj.s)
        print "  %s edges" % len(self._obj.t)
        print "  nodes attributes:"
        for cname in self._obj.s.columns :
            if cname in ("succ", "pred") :
                kind = "set(int)"
            elif cname in ("on", "off") :
                kind = "set(str)"
            else :
                column = self._obj.s[cname]
                kind = str(column.dtype)
                if kind == "object" :
                    kind = type(iter(column).next()).__name__
            print "    %s: %s" % (cname, kind)
    def _make_dtx (self) :
        "compute distances to exit (dtx)"
        print "Computing DTX"
        dtx = self._obj.dtx()
        print "  => found %s distinct values" % dtx
        return dtx
    def __setitem__ (self, name, fun, replace=False) :
        print "Computing %r on each node using %s" % (name, fun.__name__)
        count = self._obj.add(name, fun, replace)
        print "  found %s distinct values" % count
    def __getitem__ (self, key) :
        if isinstance(key, tuple) :
            key, collapse = key
        else :
            collapse = False
        if key == "dtx" :
            self.dtx
        if (key, collapse) in self._items :
            return self._items[key, collapse]
        if collapse :
            print "Collapsing %r wrt %s" % (self._obj.name, key)
        else :
            print "Merging %r wrt %s" % (self._obj.name, key)
        self._items[key,collapse] = comp = Component(self._obj.merge(key, collapse),
                                                     self.num, key)
        return comp
    def _make_graph (self, col=None, engine="neato") :
        if col is None :
            col = self.col
        if col is None :
            col = "node"
        return Graph(self._obj.graph(col, engine))

class Graph (Lazy) :
    def _make_draw (self, engine=None) :
        "show an interactive picture of the state space"
        print "Drawing graph"
        try :
            labels = (raw_input("  draw labels (default yes)? ").strip().lower()
                      or "yes").startswith("y")
        except :
            labels = True
        if engine is not None :
            self._obj.layout(engine)
        self._obj.draw(labels=labels)
        raw_input("  hit ENTER to continue")
        analyse.plt.close()
        return self
    def _make_pic (self, engine=None) :
        "save a picture of the state space"
        try :
            labels = (raw_input("  draw labels (default yes)? ").strip().lower()
                      or "yes").startswith("y")
        except :
            labels = True
        print "Enter dimension for %r" % (self._obj.savename() + ".png")
        try :
            width = int(raw_input("  width (default 1024): "))
        except :
            width = 1024
        try :
            height = int(raw_input("  height (default 768): "))
        except :
            height = 768
        if engine is not None :
            self._obj.layout(engine)
        self._obj.savepix(width, height, labels=labels)
        return self

class Ring (Graph) :
    def __init__ (self, spec, graph) :
        Graph.__init__(self, graph)
        self._spec = spec
    def __getitem__ (self, rules) :
        if not isinstance(rules, tuple) :
            rules = (rules,)
        select = []
        for name in rules :
            if name[0].upper() == "R" :
                select.append(self._spec.rules[int(name[1:]) - 1])
            else :
                select.append(self._spec.constraints[int(name[1:]) - 1])
        return self.__class__(self._spec, self._spec.ring(self._spec.init(), select))

class PetriNet (Lazy) :
    def _make_pic (self) :
        "save a picture of the Petri net (assign .engine to change layout)"
        if not hasattr(self, "engine") :
            self.engine = "neato"
        def place_attr (p, a) :
            if p.tokens :
                a["penwidth"] = 3.0
            a["label"] = str(p)
        def arc_attr (l, a) :
            del a["label"]
        def trans_attr (t, a) :
            r = t.label("rule")
            while r.parent is not None :
                r = r.parent
            a["label"] = r.name()
        self._obj.draw(os.path.join(self._obj.name, "petri-net.pdf"),
                       place_attr=place_attr, trans_attr=trans_attr, arc_attr=arc_attr,
                       engine=self.engine)
        self._obj.draw(os.path.join(self._obj.name, "petri-net.png"),
                       place_attr=place_attr, trans_attr=trans_attr, arc_attr=arc_attr,
                       engine=self.engine)
        return self

class A2I (object) :
    def __init__ (self, model) :
        self.a = model
    def __getitem__ (self, name) :
        return getattr(self.a, name)

if __name__ == "__main__" :
    if len(sys.argv) < 2 :
        print "Usage: python econet.py MODEL COMMAND...\n"
        print "Commands are dot-separated chains as follows:"
        print Model.doc()
        print """
Note that .draw / .pic / .save can be chained together and the
resulting chain is equivalent to that without them.

Note also that indexable elements (kernel and universe) can be chained
with .all to build and count all the items.
"""
        print "Required objects are built lazily only whenever they are needed,"
        print "and they are cached to avoid further rebuilds."
        sys.exit(1)
    elif not os.path.isfile(sys.argv[1]) :
        print "Could not open %r" % sys.argv[1]
        sys.exit(1)
    else :
        model = Model(sys.argv[1])
        g, l = {}, A2I(model)
        for command in sys.argv[2:] :
            print "## %s\n" % command
            try :
                eval(command, g, l)
            except KeyboardInterrupt :
                print "\n** Interrupted by user"
                break
            # except Exception as err :
            #     print "\n**", err
            print

import collections, tokenize, StringIO, os, os.path, ast, csv, socket
import subprocess, tempfile, re, time, sys
import networkx as nx
import hlr, ktz, tables, analyse
from token import tok_name

import snakes.plugins
snk = snakes.plugins.load(["gv", "labels"], "snakes.nets", "snk")

token = collections.namedtuple("token", ["type", "kind", "text", "start", "end"])

class ParseError (Exception) :
    def __init__ (self, tok, message) :
        Exception.__init__(self, " ".join(["[%s:%s]" % tok.start, message.strip()]))

class Parser (object) :
    def __init__ (self, path) :
        self._next = None
        self._lexer = self.lex(path=path)
        self._hlrparse = hlr.Parser(None)
        self.name = os.path.basename(os.path.splitext(path)[0])
        self.path = path
    def lex (self, path=None, data=None) :
        if path is not None :
            stream = open(path)
        elif data is not None :
            stream = StringIO.StringIO(data)
        else :
            raise ValueError("no input provided")
        last = None
        for typ, txt, start, end, num in tokenize.generate_tokens(stream.readline) :
            nam = tok_name[typ]
            if nam == "NEWLINE" :
                nam = "NL"
            if nam == "COMMENT" :
                continue
            elif last == nam == "NL" :
                continue
            last = nam
            yield token(typ, nam, txt, start, end)
    def next (self) :
        if self._next is None :
            self._next = self._lexer.next()
        return self._next
    def get (self) :
        self.next()
        ret = self._next
        self._next = None
        return ret
    def expect (self, values, kinds) :
        tok = self.next()
        if values and tok.text not in values :
            raise ParseError(tok, "expected %s, got %r"
                             % ("|".join(repr(v) for v in values), tok.text))
        elif kinds and tok.kind not in kinds :
            raise ParseError(tok, "expected %s, got %r (%s)"
                             % ("|".join(kinds), tok.text, tok.kind))
        self._next = None
        return tok
    def parse_decl (self) :
        name = self.expect([], ["NAME"]).text
        self.expect([":"], ["OP"])
        self.expect([], ["NL"])
        self.expect([], ["INDENT"])
        sorts = []
        while True :
            if self.next().kind == "DEDENT" :
                self.get()
                break
            elif self.next().kind == "NL" :
                self.get()
                continue
            sorts.append(self.parse_sort())
        return name, sorts
    def parse_sort (self) :
        name = self.expect([], ["NAME"]).text
        sign = self.expect(["+", "-"], ["OP"]).text
        self.expect([":"], ["OP"])
        text = []
        while self.next().kind != "NL" :
            text.append(self.get())
        self.expect([], ["NL"])
        return name, sign, " ".join(t.text for t in text)
    def parse_spec (self) :
        decls = []
        const = []
        rules = []
        rgens = []
        while True :
            tok = self.next()
            if tok.kind == "NAME" and tok.text == "rules" :
                rules = self.parse_rules()
            elif tok.kind == "NAME" and tok.text == "constraints" :
                const = self.parse_constraints()
            elif tok.kind == "NAME" and tok.text == "use" :
                rgens.append(self.parse_rgen())
            elif tok.kind == "NAME" :
                decls.append(self.parse_decl())
            elif tok.kind == "NL" :
                self.get()
            elif tok.kind == "ENDMARKER" :
                return decls, const, rules, rgens
            else :
                raise ParseError(tok, "expected 'NAME:', got %r (%s)"
                                 % (tok.text, tok.kind))
    def _parse_rules (self, section) :
        self.expect([section], ["NAME"])
        self.expect([":"], ["OP"])
        self.expect([], ["NL"])
        self.expect([], ["INDENT"])
        rules = []
        while True :
            if self.next().kind == "DEDENT" :
                self.get()
                break
            elif self.next().kind == "NL" :
                self.get()
                continue
            rules.append(self.parse_rule())
        return rules
    def parse_rules (self) :
        return self._parse_rules("rules")
    def parse_constraints (self) :
        return self._parse_rules("constraints")
    def parse_rule (self) :
        left = self.parse_side()
        self.expect([">>"], ["OP"])
        right = self.parse_side()
        self.expect([], ["NL"])
        return left, right
    def parse_side (self) :
        states = []
        while True :
            states.append((self.expect([], ["NAME"]).text,
                           self.expect(["+", "-"], ["OP"]).text))
            sep = self.next()
            if sep.kind == "NL" :
                break
            elif sep.kind == "OP" and sep.text == "," :
                self.get()
                continue
            elif sep.kind == "OP" and sep.text == ">>" :
                break
            raise ParseError(sep, "expected NL|','|'>>', got %r" % sep.text)
        return states
    def parse_rgen (self) :
        self.expect(["use"], ["NAME"])
        ptok = self.expect([], ["STRING"])
        path = ast.literal_eval(ptok.text)
        if not os.path.isfile(path) :
            _path = os.path.join(os.path.dirname(self.path), path)
            if not os.path.isfile(_path) :
                raise ParseError(ptok, "file not found %r" % path)
            path = _path
        self.expect([":"], ["OP"])
        self.expect([], ["NL"])
        self.expect([], ["INDENT"])
        rules = []
        while self.next().kind != "DEDENT" :
            line = []
            while self.next().kind not in ("NL", "ENDMARKER") :
                line.append(self.get())
            if self.next().kind == "NL" :
                self.get()
            rules.append(self._hlrparse.parse_line(line))
        dedent = self.get()
        if rules is None :
            raise ParseError(dedent, "no high-level rules given")
        return path, rules
    def parse (self) :
        meta, const, rules, rgens = self.parse_spec()
        endtok = self.expect([], ["ENDMARKER"])
        cg, rg = [], []
        for path, lst in rgens :
            for r in self._hlrparse.expand(path, lst) :
                if r.__class__.__name__ == "Constraint" :
                    cg.append(r)
                else :
                    rg.append(r)
        if not rules :
            raise ParseError(endtok, "no rules given")
        return meta, const or [], rules, cg, rg

##
##
##

class State (object) :
    def __init__ (self, name, sign) :
        self.name = name
        self.sign = sign in ("+", True)
    def __str__ (self) :
        return "".join([self.name, "+" if self.sign else "-"])
    def neg (self) :
        return self.__class__(self.name, "-" if self.sign else "+")
    def __hash__ (self) :
        return hash(str(self))
    def __eq__ (self, other) :
        try :
            return self.name == other.name and self.sign == other.sign
        except :
            return False
    def __ne__ (self, other) :
        return not self.__eq__(other)
    def __lt__ (self, other) :
        return str(self) < str(other)
    def __le__ (self, other) :
        return str(self) <= str(other)
    def __gt__ (self, other) :
        return not self.__le__(other)
    def __ge__ (self, other) :
        return not self.__lt__(other)

class Side (set) :
    def vars (self) :
        return set(s.name for s in self)
    def neg (self) :
        return self.__class__(s.neg() for s in self)
    def __add__ (self, other) :
        return self.__class__(self | set(other))
    def __div__ (self, other) :
        exclude = {s.name for s in other}
        return self.__class__(s for s in self if s.name not in exclude)
    def __hash__ (self) :
        return reduce(int.__xor__, (hash(s) for s in self),
                      hash(self.__class__.__name__))
    def __str__ (self) :
        return ", ".join(str(s) for s in sorted(self))
    def __lt__ (self, other) :
        return len(self) < len(other) and self.issubset(other)
    def __le__ (self, other) :
        return len(self) <= len(other) and self.issubset(other)
    def __gt__ (self, other) :
        return len(self) > len(other) and self.issuperset(other)
    def __ge__ (self, other) :
        return len(self) >= len(other) and self.issuperset(other)

class Rule (object) :
    _count = 0
    def __init__ (self, left, right, parent=None, constraints=[]) :
        self.left = Side(left)
        self.right = Side(right)
        self.parent = parent
        self.constraints = set(constraints)
        self.__class__._count += 1
        self.num = self._count
    def consistant (self) :
        return not ((self.left & self.left.neg())
                    or (self.right & self.right.neg()))
    def vars (self) :
        return self.left.vars() | self.right.vars()
    def norm (self) :
        return self.right + (self.left / self.right)
    def normalise (self) :
        missing = self.left / self.right
        if missing and not self.constraints :
            return self.__class__(self.left, self.right + missing, self)
        elif missing and self.constraints :
            return self.__class__(self.left, self.right + missing, self)
        else :
            return self
    def determinise (self) :
        toadd = [[]]
        diff = self.right / self.left
        if not diff :
            yield self
        else :
            for state in diff :
                toadd = ([t + [state] for t in toadd]
                         + [t + [state.neg()] for t in toadd])
            for t in toadd :
                yield self.__class__(self.left + tuple(sorted(t)), self.right, self)
    def text (self) :
        return "%s >> %s" % (self.left, self.right)
    def name (self) :
        return self.__class__.__name__[0] + str(self.num)
    def __str__ (self) :
        if self.constraints :
            return "%s/%s: %s" % (self.name(),
                                  ",".join(str(c.num) for c in self.constraints),
                                  self.text())
        else :
            return "%s: %s" % (self.name(), self.text())
    def __hash__ (self) :
        return hash((self.__class__.__name__, self.left, self.right))
    def __eq__ (self, other) :
        try :
            return (self.__class__ == other.__class__
                    and self.left == other.left
                    and self.right == other.right)
        except :
            return False
    def __ne__ (self, other) :
        return not self.__eq__(other)
    def __lt__ (self, other) :
        return (str(self.left), str(self.right)) < (str(other.left), str(other.right))
    def __le__ (self, other) :
        return (str(self.left), str(self.right)) <= (str(other.left), str(other.right))
    def __gt__ (self, other) :
        return not self.__le__(other)
    def __ge__ (self, other) :
        return not self.lt__(other)
    def __mod__ (self, other) :
        ancestors = set()
        p = self
        while p is not None :
            ancestors.add(p)
            p = p.parent
        q = other
        while q is not None :
            if q in ancestors :
                return True
            q = q.parent
        return False

class Constraint (Rule) :
    _count = 0
    def __call__ (self, rule) :
        if self in rule.constraints :
            return rule
        elif self.left <= rule.norm() :
            if not rule.parent:
                return Rule(rule.left, self.right + (rule.right / self.right),
                            rule, rule.constraints | {self})
            else :
                return Rule(rule.left, self.right + (rule.right / self.right),
                            rule.parent, rule.constraints | {self})
        else :
            return rule

##
##
##

def mtime (path) :
    return os.stat(path).st_mtime

sort = collections.namedtuple("sort", ["name", "kind", "sign", "description"])

class Spec (object) :
    def __init__ (self, meta, constraints, rules, name="data", debug=False, path=None) :
        self.meta = tuple(meta)
        self.constraints = tuple(constraints)
        self.rules = tuple(rules)
        self.name = name
        self.path = path or name
        self.pp = tables.PathPool(path, ["png", "pdf", "eps", "romeo", "gal", "pnml",
                                         "lola"])
        self.debug = debug
        self._net = None
    def log (self, *message) :
        if self.debug :
            print " ".join(str(m) for m in message)
    @classmethod
    def parse (cls, path, debug=False) :
        parser = Parser(path)
        meta, const, rules, cg, rg = parser.parse()
        return cls([sort(name, kind, sign, desc)
                    for kind, sorts in meta
                    for name, sign, desc in sorts],
                   [Constraint([State(n, s) for n, s in left],
                               [State(n, s) for n, s in right])
                    for left, right in const] + cg,
                   [Rule([State(n, s) for n, s in left],
                         [State(n, s) for n, s in right])
                    for left, right in rules] + rg,
                   parser.name, debug=debug, path=path)
    def __str__ (self) :
        return "\n".join([
            "\n".join("# %s%s (%s): %s"
                      % (sort.name, sort.sign, sort.kind, sort.description)
                      for sort in self.meta),
            "\n".join(str(rule) for rule in self.constraints) or "# no constraints",
            "\n".join(str(rule) for rule in self.rules)
        ])
    def gal (self, path) :
        with open(path, "w") as out :
            name = re.sub("[^a-z0-9]+", "", self.name, flags=re.I)
            out.write("gal %s {\n    //*** variables ***//\n" % name)
            for sort in self.meta :
                out.write("    // %s%s: %s (%s)\n"
                          % (sort.name, sort.sign, sort.description, sort.kind))
                out.write("    int %s = %s;\n" % (sort.name, int(sort.sign == "+")))
            out.write("    //*** constraints ***//\n")
            guards = []
            for const in self.constraints :
                guard = " && ".join("(%s == %s)" % (s.name, int(s.sign))
                                    for s in const.left)
                loop = " && ".join("(%s == %s)" % (s.name, int(s.sign))
                                   for s in const.right)
                out.write("    // %s\n" % const)
                out.write("    transition C%s [%s && (!(%s))] {\n"
                          % (const.num, guard, loop))
                for s in const.right :
                    out.write("        %s = %s;\n" % (s.name, int(s.sign)))
                out.write("    }\n")
                guards.append("%s && (!(%s))" % (guard, loop))
            if guards :
                prio = "(!(%s))" % " || ".join("(%s)" % g for g in guards)
            else :
                prio = "true"
            for rule in self.rules :
                guard = " && ".join("(%s == %s)" % (s.name, int(s.sign))
                                    for s in rule.left)
                loop = " && ".join("(%s == %s)" % (s.name, int(s.sign))
                                   for s in rule.right)
                out.write("    // %s\n" % rule)
                out.write("    transition R%s [%s && (!(%s)) && %s] {\n"
                          % (rule.num, guard, loop, prio))
                for s in rule.right :
                    out.write("        %s = %s;\n" % (s.name, int(s.sign)))
                out.write("    }\n")
            out.write("}\n")
    def net (self) :
        if self._net :
            return self._net
        self._net = net = snk.PetriNet(self.name)
        for sort in self.meta :
            net.add_place(snk.Place("%s+" % sort.name))
            net.add_place(snk.Place("%s-" % sort.name))
            net.place(str(State(sort.name, sort.sign))).add([snk.dot])
        seen = set()
        for priority, group in enumerate([self.constraints, self.rules]) :
            for rule in group :
                for det in rule.normalise().determinise() :
                    if det in seen or det.left == det.right :
                        continue
                    seen.add(det)
                    t = snk.Transition(det.name())
                    t.label(rule=det, priority=priority)
                    net.add_transition(t)
                    for state in det.left :
                        net.add_input(str(state), t.name, snk.Value(snk.dot))
                    for state in det.right :
                        net.add_output(str(state), t.name, snk.Value(snk.dot))
        return net
    def expand (self) :
        if self.constraints :
            old, new = set(), set(self.rules)
            while old != new :
                old, new = new, set()
                for r in old :
                    for c in self.constraints :
                        r = c(r)
                    new.add(r)
            rules = new = old
        else :
            rules = self.rules
        l2r = collections.defaultdict(set)
        for r in rules :
            l2r[r.left].add(r)
        for group in l2r.itervalues() :
            for r in list(group) :
                if any(r.right < o.right and r % o for o in group) :
                    rules.remove(r)
                    group.remove(r)
        return rules
    def init (self) :
        return set(State(s.name, s.sign) for s in self.meta)
    def ring (self, state=None, rules=None) :
        if state is None :
            state = set([State(s.name, "+") for s in self.meta]
                        + [State(s.name, "-") for s in self.meta])
        else :
            for s in self.meta :
                both = {State(s.name, "+"), State(s.name, "-")}
                if not both & state :
                    state.update(both)
        g = analyse.Graph()
        for s in self.meta :
            both = {State(s.name, "+"), State(s.name, "-")}
            if both.issubset(state) :
                g.add_node(s.name, color="#CCCCFF")
            elif State(s.name, "+") in state :
                g.add_node(s.name, color="#CCFFCC")
            else :
                g.add_node(s.name, color="#FFCCCC")
        for rule in rules or self.rules :
            for l in rule.left :
                for r in rule.right :
                    g.add_edge(l.name, r.name, rules=[], color="black")
        if hasattr(self, "_ring") :
            g.graph["layout"] = self._ring
        else :
            self._ring = g.graph["layout"] = nx.circular_layout(g)
        g.graph["colors"] = {n : g.node[n]["color"] for n in g}
        g.graph["labels"] = {n : n for n in g}
        g.graph["shapes"] = {"o" : g.nodes()}
        for n in g :
            g.nodes[n]["pos"] = self._ring[n]
            g.nodes[n]["label"] = str(n)
            g.nodes[n]["shape"] = "circle"
        for e in g.edges() :
            g.edges[e]["label"] = ""
        return g
    def draw (self, path=None) :
        if path is None :
            path = self.pp.png
        if self._net is None :
            self._net = self.net()
        def place_attr (place, attr) :
            if place.tokens :
                attr["penwidth"] = "3"
            attr["label"] = place.name
        def trans_attr (trans, attr) :
            attr["label"] = trans.name
        def arc_attr (label, attr) :
            attr["label"] = ""
        return self._net.draw(path,
                              place_attr=place_attr,
                              trans_attr=trans_attr,
                              arc_attr=arc_attr,
                              engine="twopi")
    def tina (self, path=None) :
        if self._net is None :
            self._net = self.net()
        if path is None :
            path = self.pp.net
        with open(path, "w") as out :
            out.write("net {%s}\n" % self._net.name)
            for place in self._net.place() :
                out.write("pl {%s} (%s)\n" % (place.name, len(place.tokens)))
            priority = [[], []]
            for trans in self._net.transition() :
                hist = []
                rule = trans.label("rule")
                while rule is not None :
                    hist.append(rule.name())
                    rule = rule.parent
                name = ":".join(hist)
                out.write("tr {%s} %s -> %s\n"
                          % (name,
                             " ".join("{%s}" % p for p, a in trans.input()),
                             " ".join("{%s}" % p for p, a in trans.output())))
                priority[trans.label("priority")].append(name)
            for high in priority[0] :
                for low in priority[1] :
                    out.write("pr {%s} < {%s}\n" % (low, high))
    def romeo (self, path=None) :
        if path is None :
            path = self.pp.romeo
        self.tina()
        subprocess.call(["ndrio", "-romeo", self.pp.net, path])
    def pnml (self, path=None) :
        if path is None :
            path = self.pp.pnml
        self.tina()
        subprocess.call(["ndrio", "-pnml", self.pp.net, path])
    def lola (self, path=None) :
        if path is None :
            path = self.pp.lola
        self.tina()
        subprocess.call(["ndrio", "-lola", self.pp.net, path])
    def ktz (self, path=None, progress=False) :
        if path is None :
            path = self.pp.ktz
        if ktz.needs_update([self.pp.net], [self.pp.rr]) :
            self.tina()
        start = time.time()
        if progress :
            cmd = ["-stats"]
        else :
            cmd = []
        try :
            if socket.gethostname() == "pixie" :
                cmd.append("-prclosed")
        except :
            pass
        try :
            out = subprocess.check_output(["tina"] + cmd + ["-ktz", self.pp.net, path])
        except subprocess.CalledProcessError as err :
            if os.path.exists(path) :
                os.unlink(path)
            raise SystemError("tina failed (%s)" % err.returncode)
        stats = {"path" : path,
                 "time" : time.time() - start,
                 "net" : {"name" : self._net.name,
                          "places" : len(self._net._place),
                          "transitions" : len(self._net._trans)}}
        lines = [l.strip("# ") for l in out.strip().splitlines()]
        if not (lines[0].startswith("net ")
                and (", %(places)s places, %(transitions)s transitions" % stats["net"])
                in lines[0]) :
            raise ValueError("Could not parse Tina output: %s" % out.splitlines()[0])
        for prop in lines[1].split(", ") :
            if prop.startswith("not ") :
                stats["net"][prop[4:]] = False
            else :
                stats["net"][prop] = True
        keys = lines[2].split()[1:]
        for ln in lines[3:] :
            vals = ln.split()
            s = stats[vals.pop(0)] = {}
            for k, v in zip(keys, vals) :
                s[k] = int(v)
        return stats
    def size (self) :
        with tempfile.NamedTemporaryFile() as tmp :
            self.gal(tmp.name)
            data = subprocess.check_output(["its-reach", "-i", tmp.name, "-t", "GAL",
                                            "--stats"], stderr=subprocess.STDOUT)
            states, trans = None, None
            for line in data.splitlines() :
                if line.strip().startswith("Exact state count :") :
                    states = int(line.split()[-1])
                elif line.strip().startswith("Total edges in reachability graph :") :
                    trans = int(line.split()[-1])
            return states, trans

def draw (G) :
    nx.draw(G, G.graph["layout"], nodelist=[])
    for shape, nodes in G.graph["shapes"].items() :
        colors = [G.graph["colors"][n] for n in nodes]
        nx.draw_networkx_nodes(G, G.graph["layout"],
                               nodelist=nodes,
                               node_color=colors,
                               node_size=600,
                               node_shape=shape)
    lbl = {}
    for src, tgt in G.edges() :
        try :
            lbl[src,tgt] = str(G.edge[src][tgt]["rule"].parent.num)
        except :
            lbl[src, tgt] = "+".join(sorted(set(str(r.parent.num)
                                                for r in G.edge[src][tgt]["rules"])))
    nx.draw_networkx_edge_labels(G, G.graph["layout"],
                                 edge_labels=lbl,
                                 font_size=8,
                                 label_pos=.4)
    nx.draw_networkx_labels(G, G.graph["layout"],
                            labels=G.graph["labels"],
                            with_labels=True,
                            font_size=8)

def aut2dot (source) :
    target = os.path.splitext(source)[0] + ".dot"
    E, V = set(), {}
    with open(source) as src :
        for line in src :
            line = line.strip()
            if not line :
                break
            if line.startswith("des") :
                I, T, S = ast.literal_eval(line[3:])
            else :
                f, l, t = ast.literal_eval(line)
                E.update([f, t])
                V[f,t] = l
    with open(target, "w") as tgt :
        tgt.write('digraph aut2dot {\n')
        tgt.write('  %s [label="init"];\n' % I)
        for (f, t), l in V.iteritems() :
            tgt.write('  %s -> %s [label="%s"];\n' % (f, t, l))
        tgt.write('}\n')

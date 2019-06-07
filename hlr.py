import tokenize, StringIO, collections, os.path
from token import tok_name
from snakes.data import iterate, cross
from rr2pn import Side, Rule, Constraint, State

##
## meta rules
##

class MetaRule (object) :
    def __init__ (self, left, right) :
        self.left = left
        self.right = right
    def __str__ (self) :
        return "%s >> %s" % (self.left, self.right)
    def __iter__ (self) :
        for l in self.left :
            for r in self.right :
                yield Rule(l, r)

def Cons (mr) :
    for rule in mr :
        yield Constraint(rule.left, rule.right)

class SetOfSets (object) :
    def __init__ (self, sets) :
        self.sets = frozenset(frozenset(iterate(s)) for s in iterate(sets))
    def __iter__ (self) :
        for s in self.sets :
            yield Side(s)
    def __or__ (self, other) :
        return SetOfSets(self.sets | other.sets)
    def __and__ (self, other) :
        if other is None :
            return self
        return SetOfSets(l|r for l, r in cross([self.sets, other.sets]))
    def __invert__ (self) :
        sets = [SetOfSets([m.neg()] for m in s) for s in self.sets]
        if len(sets) == 1 :
            return sets[0]
        else :
            return reduce(SetOfSets.__and__, sets)
    def __rshift__ (self, other) :
        return MetaRule(self, other)

def All (items) :
    return SetOfSets([iterate(items)])

def Any (items) :
    return SetOfSets([m] for m in iterate(items))

##
## natural syntax
##

class _Env (dict) :
    def __getitem__ (self, name) :
        if name not in self :
            self[name] = All(State(name, "+"))
        return dict.__getitem__(self, name)

class ParseError (Exception) :
    def __init__ (self, tok, message) :
        if tok is None :
            Exception.__init__(self, message)
        else :
            Exception.__init__(self, " ".join(["[%s:%s]" % tok.start, message.strip()]))

token = collections.namedtuple("token", ["type", "kind", "text", "start", "end"])

class Parser (object) :
    _verbs = {}
    @classmethod
    def rule (cls, *aliases) :
        def register (fun) :
            for a in aliases :
                cls._verbs[a] = fun
            return fun
        return register
    def __init__ (self, data) :
        if data is None :
            self._it = ()
        else :
            self._it = self.parse(data)
    def __iter__ (self) :
        for left, verb, right, cond in self._it :
            if verb not in self._verbs :
                raise ParseError("unknown action %r" % verb)
            for rule in self._verbs[verb](left, right, cond) :
                yield rule
    def parse (self, data) :
        if os.path.isfile(data) :
            data = open(data).read()
        for line in data.splitlines() :
            ret = self.parse_line(line)
            if ret is not None :
                yield ret
    def expand (self, path, rules) :
        verbs = {}
        def rule (*aliases) :
            def register (fun) :
                for a in aliases :
                    verbs[a] = fun
                return fun
            return register
        execfile(path, {"rule" : rule, "Any" : Any, "All" : All, "Cons" : Cons})
        for left, verb, right, cond in rules :
            if verb not in verbs :
                raise ParseError(None, "unknown action %r" % verb)
            for rule in verbs[verb](left, right, cond) :
                for r in rule :
                    yield r
    def parse_line (self, line) :
        if isinstance(line, (str, unicode)) :
            if not line.strip() :
                # empty line
                return
            toks = list(self.lex(line.strip()))
            if not toks :
                # only comments
                return
            self._toks = toks
        else :
            if line[-1].kind != "ENDMARKER" :
                line.extend(self.lex(""))
            self._toks = line
        self._next = 0
        return self.parse_rule()
    @classmethod
    def lex (cls, data) :
        stream = StringIO.StringIO(data)
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
    def next (self, ahead=1) :
        try :
            return self._toks[self._next + ahead - 1]
        except IndexError :
            return self._toks[-1]
    def get (self) :
        tok = self._toks[self._next]
        self._next += 1
        return tok
    def expect (self, values, kinds) :
        tok = self.next()
        if values and tok.text not in values :
            raise ParseError(tok, "expected %s, got %r"
                             % ("|".join(repr(v) for v in values), tok.text))
        elif kinds and tok.kind not in kinds :
            raise ParseError(tok, "expected %s, got %r (%s)"
                             % ("|".join(kinds), tok.text, tok.kind))
        return self.get()
    def parse_rule (self) :
        left = self.parse_side()
        verb = self.parse_verb()
        right = self.parse_side()
        cond = self.parse_cond()
        return left, verb, right, cond
    def parse_side (self) :
        side = []
        while True :
            side.append(self.expect([], ["NAME"]).text)
            if self.next().text != "," :
                return [State(s, "+") for s in side]
            self.get()
    def parse_verb (self) :
        verb = []
        nxt = self.next(2)
        while nxt.kind != "ENDMARKER" and nxt.text not in(",", "if") :
            verb.append(self.get().text)
            nxt = self.next(2)
        return " ".join(verb)
    def parse_cond (self) :
        if self.next().kind == "ENDMARKER" :
            return None
        subst = {"not" : "~",
                 "and" : "&",
                 "or" : "|"}
        self.expect(["if"], [])
        tail = self._toks[self._next:-1]
        self._next = len(self._toks) - 1
        for i, t in enumerate(tail) :
            tail[i] = subst.get(t.text, t.text)
        return eval(" ".join(tail), _Env())

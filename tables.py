import bz2, csv, os, os.path, collections
import pandas

csv.field_size_limit(4*csv.field_size_limit())

class Table (object) :
    @classmethod
    def addext (cls, path) :
        if path.endswith(".csv.bz2") :
            return path
        else :
            return path + ".csv.bz2"

class TableWriter (Table) :
    def __init__ (self, path, columns) :
        self.cols = tuple(columns)
        self.row = collections.namedtuple("row", self.cols)
        self.path = self.addext(path)
        base = os.path.dirname(self.path)
        if not os.path.exists(base) :
            os.makedirs(base)
        self._raw = bz2.BZ2File(self.path, mode="w", compresslevel=9)
        self._csv = csv.writer(self._raw)
        self.writerow(columns)
    def __enter__ (self) :
        return self
    def __exit__ (self, exc_type, exc_val, exc_tb) :
        self._raw.close()
    def writerow (self, items) :
        self._csv.writerow(items)
    def close (self) :
        self._raw.close()

class TableReader (Table) :
    @classmethod
    def dataframe (cls, path) :
        table = pandas.read_csv(cls.addext(path))
        for col in table.columns :
            table.loc[table[col].isnull(), col] = b""
        return table
    def __init__ (self, path, tmp=False) :
        self.path = self.addext(path)
        if tmp :
            self.tmp = self.path + ".tmp"
            os.rename(self.path, self.tmp)
            self._raw = bz2.BZ2File(self.tmp)
        else :
            self.tmp = None
            self._raw = bz2.BZ2File(self.path)
        self._csv = csv.reader(self._raw)
        self.cols = tuple(self._csv.next())
        self.row = collections.namedtuple("row", self.cols)
    def __enter__ (self) :
        return self
    def __exit__ (self, exc_type, exc_val, exc_tb) :
        self._raw.close()
        if self.tmp :
            os.unlink(self.tmp)
    def __iter__ (self) :
        for row in self._csv :
            yield self.row(*row)
    def close (self) :
        self._raw.close()

class PathPool (object) :
    def __init__ (self, path, extensions=[]) :
        base = path
        while True :
            base, ext = os.path.splitext(base)
            if not ext :
                break
        self.base = base
        self.name = os.path.basename(base)
        self.extensions = set(["ktz", "net", "rr"] + list(extensions))
    def __getattr__ (self, ext) :
        if ext not in self.extensions :
            raise ValueError("invalid extension %r" % ext)
        return self.base + "." + ext
    def nodes (self, **opts) :
        return self._nodes_edges("nodes", opts)
    def edges (self, **opts) :
        return self._nodes_edges("edges", opts)
    def _nodes_edges (self, kind, opts) :
        _opts = {"reduced"    : "reduced",
                 "reduce"     : "reduced",
                 "smashed"    : "smashed",
                 "smash"      : "smashed",
                 "components" : "components",
                 "compo"      : "components"}
        path = [kind]
        for name, val in sorted(opts.iteritems()) :
            if name not in _opts :
                raise TypeError("invalid argument %r" % name)
            name = _opts[name]
            if name == "components" and val :
                path.insert(0, name)
            elif val :
                path.append(name)
        return Table.addext(os.path.join(self.base, "-".join(path)))

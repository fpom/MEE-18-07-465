import colorsys, csv, collections, os, os.path, sys, bz2
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from snakes.nets import hdict, Marking, MultiSet, dot
from snakes.nets import StateGraph as OriginalStateGraph
import ktz, tables

def log (*message) :
    sys.stdout.write(" ".join(str(m) for m in message))
    sys.stdout.flush()

class StateGraph (OriginalStateGraph) :
    def build (self, progress=False) :
        for i, state in enumerate(self._build()) :
            if progress and (i % 100) == 0 :
                log("\r  %s states..." % len(self))
        log("\r")

class Graph (nx.DiGraph) :
    """Graph structure with helper methods."""
    @classmethod
    def _hexcol (cls, *rgb) :
        return "#%02X%02X%02X" % tuple(int(x * 255) for x in rgb)
    @classmethod
    def colors (cls, count):
        "Generate count distinct colors"
        # iterate over HLS color space and generate count distinct values
        # for hue and luminance so that colors can be separated even in grayscale
        return [cls._hexcol(*colorsys.hls_to_rgb(i/360., .6 + i/1200., 1.))
                for i in np.arange(0., 360., 360. / count)]
    @classmethod
    def palette (cls) :
        S = cls()
        for i in range(11) :
            S.add_node(i,
                       shape="square",
                       label="%u%%" % (i * 10),
                       color=cls._hexcol(*colorsys.hsv_to_rgb(i/10., .5 + i/20., 1)))
        for i in range(10) :
            S.add_edge(i, i+1, label="")
        S.layout("circo")
        return S
    @classmethod
    def _rules2label (cls, rules) :
        "Generate a label from a set of rules."
        lbl = set()
        for r in rules :
            while r.parent is not None :
                r = r.parent
            lbl.add(r.num)
        return "+".join(str(n) for n in sorted(lbl))
    def _nodeattr (self, attr) :
        "Return a dict {k:n} by reversing relation node[n][attr] => k}."
        data = collections.defaultdict(set)
        for node in self :
            data[self.nodes[node][attr]].add(node)
        return data
    def layout (self, engine=None) :
        """Compute a new layout for the graph resorting to a Graphviz engine
        (dot, neato, twopi, circo, fdp)."""
        engine = engine or self.graph.get("engine", "neato")
        for n in self :
            self.nodes[n].pop("pos", None)
        lo = nx.nx_pydot.pydot_layout(self, prog=engine)
        if lo is None :
            nx.nx_pydot.pydot_from_networkx(self).write_dot("debug.dot")
            raise SystemError("%s failed" % engine)
        for n, pos in lo.items() :
            self.nodes[n]["pos"] = pos
        self.graph["engine"] = engine
    def descendants (self, node, addnode=False) :
        """Return the set of nodes reachable from node, including node itself
        if addnode is True."""
        if addnode :
            return nx.descendants(self, node) | {node}
        else :
            return nx.descendants(self, node)
    def ancestors (self, node, addnode=False) :
        """Return the set of nodes from which node can be reached,
        including node itself if addnode is True."""
        if addnode :
            return nx.ancestors(self, node) | {node}
        else :
            return nx.ancestors(self, node)
    def draw (self, width=800, height=600, labels=True) :
        """Draws a the graph with Matplotlib. A window popups automatically
        if the method is called within 'ipython ---matplotlib'."""
        layout = {n : self.nodes[n]["pos"] for n in self}
        nodes_labels = {n : self.nodes[n]["label"] for n in self}
        edges_labels = {(s, d) : self.edges[s,d]["label"] for s, d in self.edges()}
        shapes = self._nodeattr("shape")
        nx.draw(self, layout, nodelist=[])
        _shapes = {"square" : "s",
                   "circle" : "o",
                   "diamond" : "D",
                   "hexagon" : "h"}
        for shape, nodes in shapes.items() :
            colors = [self.nodes[n]["color"] for n in nodes]
            nx.draw_networkx_nodes(self, layout,
                                   nodelist=nodes,
                                   node_color=colors,
                                   node_size=600,
                                   node_shape=_shapes.get(shape, "*"))
        edges = self.edges()
        colors = [self.edges[src,dst].get("color", "black") for src, dst in edges]
        nx.draw_networkx_edges(self, layout,
                               edgelist=edges,
                               edge_color=colors)
        if labels :
            nx.draw_networkx_edge_labels(self, layout,
                                         edge_labels=edges_labels,
                                         font_size=8,
                                         label_pos=.4)
            nx.draw_networkx_labels(self, layout,
                                    labels=nodes_labels,
                                    with_labels=True,
                                    font_size=8)
        win = plt.get_current_fig_manager()
        win.resize(width, height)
        plt.show(block=False)
    def savename (self, mkdir=False) :
        path = os.path.join(self.graph["basename"], self.graph["name"])
        base = os.path.dirname(path)
        if mkdir and not os.path.isdir(base) :
            os.makedirs(base)
        return path
    def savepix (self, width=800, height=600, labels=True) :
        self.draw(width, height, labels=labels)
        if labels :
            basename = self.savename()
        else :
            basename = self.savename() + "_nolabels"
        plt.savefig("%s.pdf" % basename, transparent=True, frameon=False)
        plt.savefig("%s.png" % basename, transparent=True, frameon=False, dpi=200)
        plt.close()
    def _dump (self, data) :
        if isinstance(data, (set, list, tuple, frozenset)) :
            return ", ".join(str(d) for d in sorted(data))
        elif isinstance(data, (bool, int, float)) :
            return data
        elif data is None :
            return None
        else :
            return str(data)
    def nodes_info (self) :
        raise NotImplementedError("abstract method")
    def node_dump (self, node) :
        raise NotImplementedError("abstract method")
    def edges_info (self) :
        raise NotImplementedError("abstract method")
    def edge_dump (self, src, dst) :
        raise NotImplementedError("abstract method")
    def rules_info (self) :
        yield None, "data about the rules involved in a graph"
        yield "id", "rule number"
        yield "left", "left hand side of the rule (condition)"
        yield "right", "right-hand side of the rule (outcome)"
        yield "parent", "parent of a normalised rule (empty for a model rule)"
    def rule_dump (self, rule) :
        yield rule.num
        yield rule.left
        yield rule.right
        yield rule.parent.num if rule.parent else None
    def dump (self) :
        iter_rules = self.graph["rules"].__iter__
        for tpl, info, dump, walk in [
                ("%s-rules.csv", self.rules_info, self.rule_dump, iter_rules),
                ("%s-nodes.csv", self.nodes_info, self.node_dump, self.nodes),
                ("%s-edges.csv", self.edges_info, self.edge_dump, self.edges)] :
            with open(tpl % self.savename(True), "wb") as rawout :
                out = csv.writer(rawout)
                header = []
                for name, description in info() :
                    if name is None :
                        out.writerow([description])
                        out.writerow([])
                    else :
                        out.writerow([name, description])
                        header.append(name)
                out.writerow([])
                out.writerow(header)
                for data in sorted(walk()) :
                    if isinstance(data, (tuple, list)) :
                        out.writerow([self._dump(d) for d in dump(*data)])
                    else :
                        out.writerow([self._dump(d) for d in dump(data)])

def _s2m (state, sorts) :
    bits = bin(state)[2:].rjust(len(sorts), "0")
    return Marking(("%s%s" % (s, "+" if b == "1" else "-"), MultiSet([dot]))
                   for s, b in zip(sorts, bits))

def _m2s (marking, sorts) :
    return int("".join("1" if marking(s + "+") else "0" for s in sorts), 2)

class StateSpace (Graph) :
    """A graph class to store a state space."""
    @classmethod
    def load (cls, path, engine="neato", reduced=False) :
        pp = tables.PathPool(path)
        nodes = ktz.TableReader.dataframe(pp.nodes(reduced=reduced))
        edges = ktz.TableReader.dataframe(pp.edges(reduced=reduced))
        colors = cls.colors(nodes.component.unique().size)
        S = cls(name=pp.name)
        for i, row in nodes.iterrows() :
            if row.init :
                shape = "hexagon"
            elif row.dead :
                shape = "square"
            else :
                shape = "circle"
            S.add_node(row.node,
                       label="%s/%s" % (row.node, row.component),
                       color=colors[row.component],
                       init=row.init,
                       dead=row.dead,
                       shape=shape)
        for i, row in edges.iterrows() :
            if row.src != row.dst :
                S.add_edge(row.src, row.dst,
                           label=row.rules.replace(",", "|"))
        S.layout(engine)
        return S
    @classmethod
    def build (cls, net, engine="neato", progress=False) :
        """Build a graph from a Petri net by computing its state space."""
        mg = StateGraph(net)
        mg.build(progress)
        G = cls(basename=net.name, name=net.name, rules=set())
        for state in mg :
            marking = mg.net.get_marking()
            G.add_node(state,
                       on=set(p[:-1] for p in marking if p.endswith("+")),
                       off=set(p[:-1] for p in marking if p.endswith("-")),
                       init=(state == 0),
                       dead=True)
            for succ, trans, mode in mg.successors() :
                if succ != state :
                    rules = set([trans.label("rule")])
                    G.graph["rules"].update(rules)
                    if G.has_edge(state, succ) :
                        G.edges[state,succ]["rules"].update(rules)
                        G.edges[state,succ]["label"] = cls._rules2label(G.edges[state,succ]["rules"])
                    else :
                        G.add_edge(state, succ,
                                   rules=rules,
                                   label=cls._rules2label(rules))
                    G.nodes[state]["dead"] = False
            if G.nodes[state]["dead"] :
                G.nodes[state]["shape"] = "square"
            elif G.nodes[state]["init"] :
                G.nodes[state]["shape"] = "hexagon"
            else :
                G.nodes[state]["shape"] = "circle"
        G._cleanup(engine)
        return G
    @classmethod
    def universe (cls, net, engine="neato") :
        """Build a graph from a Petri net by computing its state space."""
        G = cls(basename=net.name, name="%s-all" % net.name, rules=set())
        G.graph["sorts"] = sorts = list(set(p.name[:-1] for p in net.place()))
        for state in xrange(2**len(sorts)) :
            marking = _s2m(state, sorts)
            net.set_marking(marking)
            successors = set()
            for t in net.transition() :
                for m in t.modes() :
                    t.fire(m)
                    s = net.get_marking()
                    if s != marking :
                        successors.add((t, _m2s(s, sorts)))
                        net.set_marking(marking)
            dead = not successors
            G.add_node(state,
                       on=set(p[:-1] for p in marking if p.endswith("+")),
                       off=set(p[:-1] for p in marking if p.endswith("-")),
                       init=True,
                       dead=dead,
                       shape="square" if dead else "circle")
            for trans, succ in successors :
                rules = set([trans.label("rule")])
                if G.has_edge(state, succ) :
                    G.edges[state,succ]["rules"].update(rules)
                    G.edges[state,succ]["label"] = cls._rules2label(G.edges[state,succ]["rules"])
                else :
                    G.add_edge(state, succ,
                               rules=rules,
                               label=cls._rules2label(rules))
                G.nodes[succ]["init"] = False
        for node in (n for n in G if G.nodes[n]["init"]) :
            G.nodes[node]["shape"] = "hexagon"
        U = nx.Graph(G)
        for i, component in enumerate(nx.connected_components(U)) :
            S = G.subgraph(component).copy()
            S.graph["name"] += "-%03u" % (i + 1)
            S._cleanup(engine)
            yield S
    def _cleanup (self, engine) :
        if not self.graph["rules"] :
            for src, dst, data in self.edges(data=True) :
                self.graph["rules"].update(data["rules"])
        self.graph["rules"].update([r.parent for r in self.graph["rules"]])
        scc = self.graph["scc"] = list(nx.strongly_connected_components(self))
        scc.sort(key=min)
        colors = self.colors(len(scc))
        n2s = {}
        for num, members in enumerate(scc) :
            for node in members :
                self.nodes[node]["scc"] = num
                self.nodes[node]["color"] = colors[num]
                self.nodes[node]["label"] = "%s/%s" % (node, num)
                self.nodes[node]["members"] = set(members)
                n2s[node] = num
        for src in self.edges :
            for dst in self.edges[src] :
                self.edges[src,dst]["src_scc"] = n2s[src]
                self.edges[src,dst]["dst_scc"] = n2s[dst]
        self.layout(engine)
    def kernels (self, engine="neato") :
        """Generates the subgraphs corresponding to non-elementary SCC. Nodes
        are coloured with respect to the distance-to-exit (DTX)."""
        for num, members in enumerate(self.graph["scc"]) :
            if len(members) == 1 :
                # skip elementary SCC
                continue
            # extract subgraph
            K = KernelGraph(self.subgraph(members).copy())
            K.graph["rules"] = set(r for s in K.edges for d in K.edges[s]
                                   for r in K.edges[s,d]["rules"])
            K.graph["rules"].update([r.parent for r in K.graph["rules"]])
            K.graph["name"] = "%s-scc-%03u" % (self.graph["name"], num)
            K.graph["basename"] = self.graph["basename"]
            # order nodes by DTX
            # DTX == 0 => all the nodes with an exit out of the SCC
            layer = set(src for src in K for dst in self.edges[src]
                        if self.nodes[dst]["scc"] != num)
            known = set()
            dtx = 0
            while layer :
                for node in layer :
                    K.nodes[node]["dtx"] = dtx
                    K.nodes[node]["members"] = layer
                known.update(layer)
                # compute next layer with DTX + 1
                pred = reduce(set.union, (set(K.pred[node])
                                          for node in layer)) - known
                layer = pred
                dtx += 1
            # build colors, labels and shapes wrt DTX
            colors = self.colors(dtx)
            for node in K :
                K.nodes[node]["color"] = colors[K.nodes[node]["dtx"]]
                K.nodes[node]["label"] = "%s/%s" % (node, K.nodes[node]["dtx"])
                if [p for p in self.pred[node] if self.nodes[p]["scc"] != num] :
                    K.nodes[node]["init"] = True
                    K.nodes[node]["shape"] = "hexagon"
                else :
                    K.nodes[node]["init"] = False
                    K.nodes[node]["shape"] = "circle"
            K.layout(engine)
            yield K
    def scc (self) :
        "Build the components (SCC + basins) graph"
        done, comp, todo = set(), set(), set()
        for node, data in self.nodes(data=True) :
            scc = self.graph["scc"][data["scc"]]
            if data["init"] :
                data["comp"] = {node}
                data["kind"] = "init"
                done.add(node)
                comp.add(frozenset([node]))
                todo.update(self.predecessors(node))
            elif len(scc) > 1 :
                data["comp"] = scc
                data["kind"] = "scc"
                done.add(node)
                comp.add(frozenset(scc))
                todo.update(self.predecessors(node))
            elif data["dead"] :
                data["comp"] = {node}
                data["kind"] = "deadlock"
                done.add(node)
                comp.add(frozenset([node]))
                todo.update(self.predecessors(node))
        todo.difference_update(done)
        basins = collections.defaultdict(set)
        while todo :
            for node in todo :
                c = reduce(set.__or__,
                           (self.nodes[p].get("comp", set())
                            for p in self.successors(node)),
                           set())
                self.nodes[node]["comp"] = c
                self.nodes[node]["kind"] = "basin"
                basins[frozenset(c)].add(node)
            done.update(todo)
            todo = set(n for d in todo for n in self.predecessors(d)) - done
        comp.update(frozenset(b) for b in basins.values())
        def label (num, members) :
            if len(members) == 1 :
                return "%s/%s" % (iter(members).next(), num)
            else :
                return "(%s)" % num
        def attrs (num, members) :
            node = iter(members).next()
            return {"shape" : {"deadlock" : "square",
                               "scc" : "circle",
                               "basin" : "diamond",
                               "init" : "hexagon"}.get(self.nodes[node]["kind"], "*"),
                    "comp" : num,
                    "kind" : self.nodes[node]["kind"]}
        return SCCGraph.build(self, list(comp), label, attrs, "components")
    def nodes_info (self) :
        yield None, "data about the nodes of a state graph"
        yield "node", "state number"
        yield "succ", "the successors of this state"
        yield "pred", "the predecessors of this state"
        yield "scc", "number of the SCC this node belongs to"
        yield "on", "the components ON in this state"
        yield "off", "the components OFF in this state"
        yield "init", "whether this is the initial state"
        yield "dead", "whether this is a deadlock"
        yield "members", "the other nodes in the same SCC"
    def node_dump (self, node) :
        yield node
        yield self.successors(node)
        yield self.predecessors(node)
        yield self.nodes[node]["scc"]
        yield self.nodes[node]["on"]
        yield self.nodes[node]["off"]
        yield self.nodes[node]["init"]
        yield self.nodes[node]["dead"]
        yield self.nodes[node]["members"]
    def edges_info (self) :
        yield None, "data about the edges of a state graph"
        yield "src", "number of the source node"
        yield "dst", "number of the destination node"
        yield "rule", "number of the rule executed on the edge"
        yield "normalised", "number of the normalised rule executed on the edge"
        yield "src_scc", "number of the SCC src belongs to"
        yield "dst_scc", "number of the SCC dst belongs to"
        yield "src_members", "nodes in the same SCC than src"
        yield "dst_members", "nodes in the same SCC than dst"
    def edge_dump (self, src, dst) :
        yield src
        yield dst
        yield set(r.parent.num for r in self.edges[src,dst]["rules"])
        yield set(r.num for r in self.edges[src,dst]["rules"])
        yield self.nodes[src]["scc"]
        yield self.nodes[dst]["scc"]
        yield self.nodes[src]["members"]
        yield self.nodes[dst]["members"]

class KernelGraph (Graph) :
    """A graph class to store a SCC of a state space."""
    def reduce (self, collapse=False) :
        """Merge the nodes that avec the same DTX (distance to exit)
         - always if collapse is True
         - only if they are SCC themselves otherwise
        """
        if collapse :
            classes = [v for k, v in sorted(self._nodeattr("dtx").items())]
            name = "dtx-min"
            def label (num, members) :
                return "(%s)" % num
            def attrs (num, members) :
                return {"dtx" : self.nodes[iter(members).next()]["dtx"],
                        "shape" : "hexagon" if any(self.nodes[m]["init"]
                                                   for m in members) else "circle"}
        else :
            classes = []
            for nodes in self._nodeattr("dtx").values() :
                if nx.is_strongly_connected(self.subgraph(nodes)) :
                    classes.append(nodes)
                else :
                    classes.extend({n} for n in nodes)
            classes.sort(key=min)
            name = "dtx-scc"
            def label (num, members) :
                if len(members) == 1 :
                    first = iter(members).next()
                    return "%s/%s" % (first, self.nodes[first]["dtx"])
                else :
                    return "(%s)" % num
            def attrs (num, members) :
                return {"dtx" : self.nodes[iter(members).next()]["dtx"],
                        "shape" : "hexagon" if any(self.nodes[m]["init"]
                                                   for m in members) else "circle"}
        return ReducedKernelGraph.build(self, classes, label, attrs, name, "circo")
    def nodes_info (self) :
        yield None, "data about the nodes of a SCC of a state graph (kernel)"
        yield "node", "state number"
        yield "succ", "the successors of this state within the SCC"
        yield "pred", "the predecessors of this state within the SCC"
        yield "scc", "the SCC this node belongs to"
        yield "dtx", "distance from this node to an exit of the SCC (DTX)"
        yield "on", "the components ON in this state"
        yield "off", "the components OFF in this state"
        yield "init", "whether we can enter the SCC through this node"
        yield "dead", "whether this is a deadlock"
        yield "members", "the other nodes with the same DTX"
    def node_dump (self, node) :
        yield node
        yield self.successors(node)
        yield self.predecessors(node)
        yield self.nodes[node]["scc"]
        yield self.nodes[node]["dtx"]
        yield self.nodes[node]["on"]
        yield self.nodes[node]["off"]
        yield self.nodes[node]["init"]
        yield self.nodes[node]["dead"]
        yield self.nodes[node]["members"]
    def edges_info (self) :
        yield None, "data about the edges of a SCC of a state graph"
        yield "src", "source node"
        yield "dst", "destination node"
        yield "rule", "rule executed on the edge"
        yield "normalised", "normalised rule executed on the edge"
        yield "src_dtx", "DTX of src"
        yield "dst_dtx", "DTX of dst"
        yield "src_members", "nodes with the same DTX than src"
        yield "dst_members", "nodes with the same DTX than dst"
    def edge_dump (self, src, dst) :
        yield src
        yield dst
        yield set(r.parent.num for r in self.edges[src,dst]["rules"])
        yield set(r.num for r in self.edges[src,dst]["rules"])
        yield self.nodes[src]["dtx"]
        yield self.nodes[dst]["dtx"]
        yield self.nodes[src]["members"]
        yield self.nodes[dst]["members"]

class MergedGraph (Graph) :
    """A graph class to store graphs with merged nodes."""
    @classmethod
    def load (cls, path, engine="neato", reduced=False) :
        pp = tables.PathPool(path)
        nodes = ktz.TableReader.dataframe(pp.nodes(components=True, reduced=reduced))
        edges = ktz.TableReader.dataframe(pp.edges(components=True, reduced=reduced))
        S = cls(basename=pp.name,
                name=pp.name + "-components")
        size = float(nodes["size"].sum())
        shapes = {"INIT" : "hexagon",
                  "SCCINIT" : "hexagon",
                  "DEAD" : "square",
                  "SCC" : "circle",
                  "BASIN" : "diamond"}
        for i, row in nodes.iterrows() :
            S.add_node(row.node,
                       label="(%s)" % row.node,
                       color=cls._hexcol(*colorsys.hsv_to_rgb(row["size"]/size,
                                                              .5 + row["size"]/size/2,
                                                              1)),
                       init="INIT" in row.kind,
                       dead=row.kind == "DEAD",
                       shape=shapes[row.kind])
        for i, row in edges.iterrows() :
            if row.src != row.dst :
                S.add_edge(row.src, row.dst,
                           label=row.rules.replace(",", "|"))
        S.layout(engine)
        return S
    @classmethod
    def build (cls, G, classes, label, attrs, name, engine="dot") :
        """Build a graph from a StateSpace instance."""
        if not isinstance(G, (StateSpace, KernelGraph)) :
            raise ValueError("graph has wrong type (%s)" % G.__class__.__name__)
        S = cls(basename=G.graph["basename"],
                name="%s-%s" % (G.name, name),
                rules=set(G.graph["rules"]))
        klass = {}
        for num, members in enumerate(classes) :
            klass.update({m : num for m in members})
            S.add_node(num,
                       members=members,
                       label=label(num, members),
                       color=G.nodes[iter(members).next()]["color"],
                       init=any(G.nodes[n]["init"] for n in members),
                       dead=all(G.nodes[n]["dead"] for n in members),
                       rules=set(),
                       off=reduce(set.intersection,
                                  (G.nodes[n]["off"] for n in members)),
                       on=reduce(set.intersection,
                                 (G.nodes[n]["on"] for n in members)),
                       **attrs(num, members))
        edges = collections.defaultdict(set)
        for src, dst in G.edges() :
            edges[klass[src], klass[dst]].update(G.edges[src,dst]["rules"])
        for (src, dst), rules in edges.items() :
            if src == dst :
                S.nodes[src]["rules"] = set(rules)
            else :
                S.add_edge(src, dst,
                           rules=set(rules),
                           label=cls._rules2label(rules),
                           src_members=set(classes[src]),
                           dst_members=set(classes[dst]))
        S.layout(engine)
        return S

def strip (paths) :
    paths = list(paths)
    for d in "><" :
        for i, n in enumerate(paths[0]) :
            if any(p[i] != n for p in paths[1:]) :
                paths = [p[i:][::-1] for p in paths]
                break
    return set(paths)

class SCCGraph (MergedGraph) :
    """A graph to store the SCC graph of a state space."""
    def diamonds (self) :
        """Compute a copy of the graph from which diamonds have been reduced."""
        S = self.copy()
        S.graph["basename"] = self.graph["basename"]
        S.graph["name"] = "%s-min" % self.graph["name"]
        deadlocks = set(n for n in S if S.nodes[n]["dead"])
        # list nodes from the initial states in breadth-first order
        # and call sorted() to make the algorithm deterministic
        candidates = list(sorted(n for n in S if S.nodes[n]["init"]))
        done = front = set(candidates)
        while front :
            front = set(n for f in front for n in S.successors(f)) - done
            candidates.extend(sorted(front))
            done.update(front)
        # diamonds reduction starts here
        while candidates :
            # process nodes in order from the initial state
            root = candidates.pop(0)
            for dead in sorted(S.descendants(root) & deadlocks) :
                # process the paths from root to each deadlock
                # reachable from it: collect sets of paths that use
                # exactly the same rules
                matches = collections.defaultdict(list)
                for path in sorted(nx.all_simple_paths(S, root, dead)) :
                    count = collections.defaultdict(int)
                    for s, d in zip(path, path[1:]) :
                        for r in S.edges[s,d]["rules"] :
                            count[r.parent.num] += 1
                    matches[hdict(count)].append(tuple(path))
                # only consider sets with more than one path
                matches = [paths for paths in sorted(matches.values(), key=min)
                           if len(paths) > 1]
                # collect all the nodes to be removed
                drop = set()
                keep = set()
                for paths in matches :
                    remaining = set(paths)
                    diamond = reduce(set.union, (set(p) for p in paths))
                    for path in paths :
                        for node in path[1:-1] :
                            if not (set(S.pred[node]) - diamond) :
                                # only remove nodes that have not predecessors
                                # out of the diamond
                                drop.add(node)
                                remaining.discard(path)
                    # preserve one path if all have been removed
                    if not remaining :
                        # choose the smallest one in lexicographic order
                        keep.update(min(paths))
                drop.difference_update(keep)
                if drop :
                    # actually remove nodes
                    S.remove_nodes_from(drop)
                    # filter out removed nodes from candidates
                    candidates = [n for n in candidates if n in S]
        S.layout()
        return S
    def nodes_info (self) :
        yield None, "data about the nodes of a component graph"
        yield "node", "component number"
        yield "succ", "the successors of this component"
        yield "pred", "the predecessors of this component"
        yield "on", "the components that are always ON in every state of this component"
        yield "off", "the components that are always OFF in every state of this component"
        yield "kind", "the kind of component this node is (scc, deadlock, basin, init)"
        yield "init", "whether the initial state is in this component"
        yield "dead", "whether this is a deadlock (thus, component reduced to a single state)"
        yield "members", "the states in this component"
    def node_dump (self, node) :
        yield node
        yield self.successors(node)
        yield self.predecessors(node)
        yield self.nodes[node]["on"]
        yield self.nodes[node]["off"]
        yield self.nodes[node]["kind"]
        yield self.nodes[node]["init"]
        yield self.nodes[node]["dead"]
        yield self.nodes[node]["members"]
    def edges_info (self) :
        yield None, "data about the edges of a component graph"
        yield "src", "source node"
        yield "dst", "destination node"
        yield "rules", "rules that can be executed on an edge from src to dst"
        yield "normalised", "the actual normalised rules"
        yield "src_members", "states in component src"
        yield "dst_members", "states in component dst"
    def edge_dump (self, src, dst) :
        yield src
        yield dst
        yield set(r.parent.num for r in self.edges[src,dst]["rules"])
        yield set(r.num for r in self.edges[src,dst]["rules"])
        yield self.nodes[src]["members"]
        yield self.nodes[dst]["members"]

class ReducedKernelGraph (MergedGraph) :
    """A graph to store the reduction of a kernet graph."""
    def nodes_info (self) :
        yield None, "data about the nodes of a reduced kernel"
        yield "node", "node number"
        yield "succ", "the successors of this node"
        yield "pred", "the predecessors of this node"
        yield "dtx", "the distance to exit (DTX) of this node"
        yield "on", "the components that are always ON in every state of this node"
        yield "off", "the components that are always OFF in every state of this node"
        yield "init", "whether the initial state is in this node"
        yield "dead", "whether this is a deadlock"
        yield "members", "the states in this node"
    def node_dump (self, node) :
        yield node
        yield self.successors(node)
        yield self.predecessors(node)
        yield self.nodes[node]["dtx"]
        yield self.nodes[node]["on"]
        yield self.nodes[node]["off"]
        yield self.nodes[node]["init"]
        yield self.nodes[node]["dead"]
        yield self.nodes[node]["members"]
    def edges_info (self) :
        yield None, "data about the edges of a reduced kernel"
        yield "src", "source node"
        yield "dst", "destination node"
        yield "rules", "rules that can be executed on an edge from src to dst"
        yield "normalised", "the actual normalised rules"
        yield "src_dtx", "DTX of src"
        yield "dst_dtx", "DTX of dst"
        yield "src_members", "states in SCC src"
        yield "dst_members", "states in SCC dst"
    def edge_dump (self, src, dst) :
        yield src
        yield dst
        yield set(r.parent.num for r in self.edges[src,dst]["rules"])
        yield set(r.num for r in self.edges[src,dst]["rules"])
        yield self.nodes[src]["dtx"]
        yield self.nodes[dst]["dtx"]
        yield self.nodes[src]["members"]
        yield self.nodes[dst]["members"]

##
## Components analysis
##

class ComponentSet (object) :
    def __init__ (self, path, reduced=False) :
        self.pp = tables.PathPool(path)
        self.c = tables.TableReader.dataframe(self.pp.nodes(components=True,
                                                            reduced=reduced))
        self.s = tables.TableReader.dataframe(self.pp.nodes(reduced=reduced))
        self.t = tables.TableReader.dataframe(self.pp.edges(reduced=reduced))
    def __len__ (self) :
        return len(self.c)
    def __iter__ (self) :
        for i in range(len(self)) :
            yield self[i]
    def ls (self) :
        l = []
        for i in range(len(self)) :
            l.append(len(self.s[self.s.component == i]))
        return l
    def __getitem__ (self, num) :
        try :
            c = self.c[self.c.node == num].iterrows().next()[1]
        except :
            raise ValueError("non-existing component")
        s = self.s[self.s.component == num]
        nodes = set(s.node)
        t = self.t[self.t.src.isin(nodes) & self.t.dst.isin(nodes)]
        s.loc[s.index, "input"] = s.node.isin(self.t[~self.t.src.isin(nodes)
                                                     & self.t.dst.isin(nodes)].dst)
        s.loc[s.index, "output"] = s.node.isin(self.t[self.t.src.isin(nodes)
                                                      & ~self.t.dst.isin(nodes)].src)
        return Component("%s-%s" % (self.pp.name, num), s, t, c.kind, self.pp)

class Component (object) :
    def __init__ (self, name, states, trans, kind, pp) :
        self.pp = pp
        self.name = name
        self.s = states
        self.t = trans
        self.kind = kind
    def __len__ (self) :
        return len(self.s)
    def _wrap (self, fun) :
        def wrapper (row) :
            new = dict(zip(row._fields, row))
            new.pop("Index")
            new["succ"] = set(int(s) for s in row.succ.split(","))
            new["pred"] = set(int(s) for s in row.pred.split(","))
            new["on"] = set(row.on.split(","))
            new["off"] = set(row.off.split(","))
            return fun(**new)
        return wrapper
    def save (self) :
        nodes = tables.Table.addext(os.path.join(self.pp.base, "nodes-" + self.name))
        with bz2.BZ2File(nodes, mode="w", compresslevel=9) as out :
            self.s.to_csv(out, index=False)
        edges = tables.Table.addext(os.path.join(self.pp.base, "edges-" + self.name))
        with bz2.BZ2File(edges, mode="w", compresslevel=9) as out :
            self.t.to_csv(out, index=False)
        return nodes, edges
    def components (self) :
        row = self.s.itertuples().next()
        return (set(row.on.split(",")) | set(row.off.split(","))) - set([""])
    def add (self, col, fun, replace=False) :
        if col in self.s.columns and not replace :
            raise ValueError("column %r exists already" % col)
        self.s.loc[self.s.index, col] = map(self._wrap(fun), self.s.itertuples())
        return len(self.s[col].unique())
    def dtx (self) :
        done = set(self.s[self.s.output == True].node)
        if not done :
            self.s.loc[self.s.index, "dtx"] = -1
        else :
            self.s.loc[self.s.node.isin(done), "dtx"] = 0
            todo = set(self.s[self.s.output == False].node)
            d = 1
            while todo :
                pred = set(self.t[self.t.src.isin(todo) & self.t.dst.isin(done)].src)
                self.s.loc[self.s.node.isin(pred), "dtx"] = d
                done.update(pred)
                todo.difference_update(pred)
                d += 1
        self.s.loc[self.s.index, "dtx"] = self.s.dtx.apply(int)
        return d - 1
    def merge (self, col, collapse=False) :
        if collapse :
            classes = [set(self.s[self.s[col] == c].node) for c in self.s[col].unique()]
        else :
            classes = []
            G = Graph()
            G.add_nodes_from(self.s.node)
            G.add_edges_from((row.src, row.dst) for row in self.t.itertuples())
            for c in self.s[col].unique() :
                S = G.subgraph(self.s[self.s[col] == c].node)
                classes.extend(nx.strongly_connected_components(S))
        states, trans = ktz.merge_classes(self.s, self.t, classes, col)
        return self.__class__("%s-%s" % (self.name, col),
                              states,
                              trans,
                              "%s/%s" % (self.kind, col),
                              self.pp)
    def graph (self, col, engine="neato") :
        vals = set(self.s[col].unique())
        colors = dict(zip(sorted(vals), Graph.colors(len(vals))))
        G = Graph(name=self.name + "-" + col)
        def _shape (row) :
            if row.init :
                return "hexagon"
            elif row.dead :
                return "square"
            else :
                return "circle"
        def _label (row) :
            return "%s/%s" % (row.node, getattr(row, col))
        for row in self.s.itertuples() :
            G.add_node(row.node,
                       label=_label(row),
                       color=colors[getattr(row, col)],
                       shape=_shape(row))
        for row in self.t.itertuples() :
            if row.src != row.dst :
                G.add_edge(row.src, row.dst, label=row.rules.replace(",", "|"))
        G.layout(engine)
        return G

##
## draft
##

def hysteresis (g) :
    src = dst = None
    for node in g :
        if src is None or len(g.nodes[node]["on"]) < len(g.nodes[src]["on"]) :
            src = node
        if dst is None or len(g.nodes[node]["on"]) > len(g.nodes[dst]["on"]) :
            dst = node
    num = 0
    for forward in nx.all_simple_paths(g, src, dst) :
        forward_nodes = set(forward[1:-1])
        for backward in nx.all_simple_paths(g, dst, src) :
            backward_nodes = set(backward[1:-1])
            if forward_nodes & backward_nodes :
                continue
            f = [len(g.nodes[n]["on"]) for n in forward]
            b = [len(g.nodes[n]["on"]) for n in reversed(backward)]
            title = "%s / %s" % ("-".join(str(n) for n in forward),
                                 "-".join(str(n) for n in backward))
            f.extend(None for i in range(len(b) - len(f)))
            b.extend(None for i in range(len(f) - len(b)))
            frame = pd.DataFrame({"forward" : f, "backward" : b})
            plot = frame.plot(title=title)
            plot.get_figure().savefig("plot-%s-%03u.pdf" % (g.name, num))
            num += 1

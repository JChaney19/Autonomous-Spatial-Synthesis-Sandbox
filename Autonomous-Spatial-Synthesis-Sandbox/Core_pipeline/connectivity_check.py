"""
core_pipeline/connectivity_check.py
===================================
Connectivity layer: is the model one buildable piece?

* DisjointSet (union-find with path halving)
* StudGrid  - grid-level stud tracking used while *generating* a model
  (occupancy + "which part owns the stud under this cell"), from the voxel builder
* studs_dir - every stud of a part with its pointing direction (handles SNOT / brackets)
* stud-probe connectivity on a finished .ldr, plus joints for legal intersections
  (bar in clip, axle in hole ...) supplied by telemetry_logger.is_exempt
* sub-assembly validation: "is this removable top one piece, and does the rest stay
  one piece without it?"
"""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import telemetry_logger as tl   # noqa: E402


# ---------------------------------------------------------------- union-find
class DisjointSet:
    def __init__(self, n: int = 0):
        self.parent = list(range(n))

    def add(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> dict:
        g = collections.defaultdict(list)
        for i in range(len(self.parent)):
            g[self.find(i)].append(i)
        return g


# ---------------------------------------------------------------- grid stud tracking (generation time)
class StudGrid:
    """
    Stud-grid bookkeeping for procedurally placed parts.
    Cells are (i, j, L): stud column i, stud row j, plate level L (L grows upward).
    occ[(i,j,L)]  -> part index occupying the cell
    stud[(i,j,L)] -> part index whose top stud sits on top of level L at (i,j)
    A part placed on level L joins every part owning a stud at (i,j,L-1) under its footprint,
    and every part already sitting on its own top studs (under-hung parts).
    """

    def __init__(self):
        self.occ, self.stud, self.parts = {}, {}, []
        self.dsu = DisjointSet()

    def free(self, cells, L, h) -> bool:
        return all((a, b, l) not in self.occ for a, b in cells for l in range(L, L + h))

    def add(self, cells, studs, L, h, payload=None) -> int:
        """cells/studs: lists of (i,j) absolute; h: height in plates."""
        if not self.free(cells, L, h):
            raise ValueError(f"cell clash at level {L}: {cells}")
        idx = self.dsu.add()
        for a, b in cells:
            for l in range(L, L + h):
                self.occ[(a, b, l)] = idx
        for a, b in cells:                                   # sits on studs below
            q = self.stud.get((a, b, L - 1))
            if q is not None:
                self.dsu.union(q, idx)
        top = L + h - 1
        for a, b in studs:
            self.stud[(a, b, top)] = idx
            q = self.occ.get((a, b, top + 1))                # something already above
            if q is not None and q != idx:
                self.dsu.union(q, idx)
        self.parts.append(dict(cells=cells, studs=studs, L=L, h=h, payload=payload))
        return idx

    def components(self) -> dict:
        return self.dsu.groups()

    def main_component(self) -> set:
        g = self.components()
        return set(max(g.values(), key=len)) if g else set()

    def components_without(self, skip: set) -> int:
        """Number of components if the parts in `skip` were removed (used before borrowing parts)."""
        tmp = StudGrid()
        for k, p in enumerate(self.parts):
            if k not in skip:
                tmp.add(p["cells"], p["studs"], p["L"], p["h"], p["payload"])
        return len(tmp.components())


# ---------------------------------------------------------------- directional studs (finished models)
_STUD_CACHE: dict = {}
_PRIMITIVE_PREFIXES = ("box", "rect", "con", "ring", "edge", "4-", "1-", "2-", "3-", "ndis", "disc",
                       "tang", "chrd", "torus", "rin", "8-", "48", "stud")


def studs_dir(pid: str):
    """All studs of a part as (local origin, local pointing direction); includes sideways (SNOT) studs."""
    if pid in _STUD_CACHE:
        return _STUD_CACHE[pid]
    out = []

    def rec(name, T):
        text = tl.fetch(name)
        if text is None:
            return
        for line in text.splitlines():
            p = line.split()
            if len(p) >= 15 and p[0] == "1":
                v = list(map(float, p[2:14]))
                sub = " ".join(p[14:]).lower()
                base = sub.replace("\\", "/").split("/")[-1]
                nt = tl.compose(T, [v[3:6], v[6:9], v[9:12]], v[0:3])
                if base.startswith("stud") and not base.startswith(tl.UNDERSIDE_STUDS):
                    o = tl.apply(nt, (0, 0, 0))
                    u = tl.apply(nt, (0, -1, 0))
                    out.append((o, tuple(u[k] - o[k] for k in range(3))))
                elif base.startswith("stug") or not base.startswith(_PRIMITIVE_PREFIXES):
                    rec(sub, nt)

    rec(pid + ".dat", None)
    _STUD_CACHE[pid] = out
    return out


def build_graph(parts, world=None, rules: tl.Rules = tl.DEFAULT_RULES, tol: tl.Tolerances = tl.DEFAULT_TOL):
    """
    Union parts that are physically joined:
      1. a stud joins the first part containing the point `stud_probe` LDU along its axis;
      2. legal intersections (bar in clip, axle in hole, ...) count as joints.
    Returns (DisjointSet, number of stud links, number of joint links).
    """
    world = world or [tl.world_bbox(p) for p in parts]
    dsu = DisjointSet(len(parts))
    stud_links = joint_links = 0
    for k, pl in enumerate(parts):
        for o, d in studs_dir(pl.pid):
            po = tl.mat9_apply(pl.m9, pl.pos, o)
            dd = tl.mat9_dir(pl.m9, d)
            q = tuple(po[i] + tol.stud_probe * dd[i] for i in range(3))
            for o2 in range(len(parts)):
                if o2 == k:
                    continue
                bb = world[o2][0]
                if all(bb[i] - tol.probe_eps <= q[i] <= bb[i + 3] + tol.probe_eps for i in range(3)):
                    dsu.union(k, o2)
                    stud_links += 1
                    break
    for a in range(len(parts)):
        for b in range(a + 1, len(parts)):
            if tl.is_exempt(parts[a], parts[b], rules, tol):
                if all(d > tol.overlap_eps for d in tl.overlap_depths(world[a][0], world[b][0])):
                    dsu.union(a, b)
                    joint_links += 1
    return dsu, stud_links, joint_links


def component_report(parts, dsu: DisjointSet, small: int = 10) -> dict:
    groups = dsu.groups()
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    main = max(groups.values(), key=len) if groups else []
    loose = [dict(pid=parts[k].pid, color=parts[k].color, line=parts[k].line,
                  pos=[round(v) for v in parts[k].pos])
             for g in groups.values() if len(g) < small for k in g]
    return dict(components=len(groups), sizes=sizes[:6], main_size=len(main), loose=loose[:20])


# ---------------------------------------------------------------- sub-assembly validation
def step_ranges(ldr_path: str):
    """Map each '0 // ===== STEP n: title' header to the set of part line numbers under it."""
    out, cur = collections.OrderedDict(), None
    for n, line in enumerate(open(ldr_path), 1):
        if line.startswith("0 // ===== STEP"):
            cur = line.split(":", 1)[1].strip()
            out[cur] = set()
        elif line.startswith("1 ") and cur is not None:
            out[cur].add(n)
    return out


def validate_subassembly(ldr_path: str, step_keyword: str, rules=tl.DEFAULT_RULES, tol=tl.DEFAULT_TOL) -> dict:
    """
    A removable section (e.g. every step whose title contains 'removable top') must
    (a) be one connected piece on its own, and (b) leave the rest of the model one piece.
    """
    parts = tl.load_ldr(ldr_path)
    lines = set()
    for title, ls in step_ranges(ldr_path).items():
        if step_keyword.lower() in title.lower():
            lines |= ls
    sub = [p for p in parts if p.line in lines]
    rest = [p for p in parts if p.line not in lines]
    res = dict(keyword=step_keyword, sub_parts=len(sub), rest_parts=len(rest))
    for name, group in (("sub", sub), ("rest", rest)):
        if group:
            dsu, _, _ = build_graph(group, None, rules, tol)
            res[name + "_components"] = len(dsu.groups())
    res["ok"] = res.get("sub_components", 0) == 1 and res.get("rest_components", 0) == 1
    return res


# ---------------------------------------------------------------- full check
def check(ldr_path: str, removable_keyword: str | None = None, rules=tl.DEFAULT_RULES,
          tol=tl.DEFAULT_TOL, logger: tl.TelemetryLogger | None = None) -> dict:
    """Geometry (overlaps) + connectivity (+ optional sub-assembly) report for one model."""
    logger = logger or tl.TelemetryLogger(run_name=os.path.basename(ldr_path))
    tl.geometry_report(ldr_path, rules, tol, logger)
    parts = tl.load_ldr(ldr_path)
    logger.start("connectivity")
    dsu, s_links, j_links = build_graph(parts, None, rules, tol)
    rep = component_report(parts, dsu)
    logger.stop("connectivity")
    logger.metric("stud_links", s_links)
    logger.metric("joint_links", j_links)
    for k, v in rep.items():
        logger.metric(k, v)
    if removable_keyword:
        logger.metric("subassembly", validate_subassembly(ldr_path, removable_keyword, rules, tol))
    return logger.flush()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Check an LDraw model for clashes and connectivity.")
    ap.add_argument("ldr")
    ap.add_argument("--removable", help="step-title keyword of a removable sub-assembly, e.g. 'removable top'")
    a = ap.parse_args()
    m = check(a.ldr, a.removable)["metrics"]
    print(json.dumps({k: v for k, v in m.items() if k not in ("tolerances",)}, indent=1))

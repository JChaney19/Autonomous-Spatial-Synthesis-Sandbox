"""
core_pipeline/voxel_generator.py
================================
Voxel-to-brick generator (the Hammerhead corvette pipeline).

    shape (voxels + colour preferences)
      -> catalog / part table (what every inventory part is, in grid terms)
      -> Builder.fill   (bottom-up greedy brick placement with stud support)
      -> Builder.prune  (keep the main connected piece)
      -> decorate       (slopes, ridges, knobs, corner tiles, tiles on exposed studs)
      -> emit_ldr       (one STEP per plate layer)

Grid: a cell is (i, j, L) = (stud along X, stud along Z, plate level). Level 0 is the
bottom; LDraw y of a cell's top = -8*(L+1). Bow of the ship is -Z.

Depends on telemetry_logger (LDraw access, part geometry) and connectivity_check
(DisjointSet). Run as a script to rebuild a ship and check it:
    python core_pipeline/voxel_generator.py rebrickable_parts.csv --preset v6 --out ship.ldr
"""
from __future__ import annotations

import collections
import csv
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import telemetry_logger as tl          # noqa: E402
from connectivity_check import DisjointSet   # noqa: E402

# ================================================================ inventory + colours
RB2LD = {41: 43, 45: 37, 182: 57, 326: 330, 1059: 365, 1095: 40, 1136: 402, 1103: 148}


def ldcolor(c: int) -> int:
    return RB2LD.get(c, c)


def load_inventory(csv_path: str) -> collections.Counter:
    """Rebrickable CSV (Part, Color or Color ID, Quantity) -> Counter[(part, rebrickable colour)]."""
    inv = collections.Counter()
    for r in csv.DictReader(open(csv_path)):
        inv[(r["Part"], int(r.get("Color ID") or r.get("Color")))] += int(r["Quantity"])
    return inv


# ================================================================ catalog (part classification)
MOVED = {"2654": "2654a", "3660": "3660a", "3665": "3665a"}       # renamed in the LDraw library
CORNER = {"2420", "2357", "77844"}                                 # L-shaped corner parts
FILLINT_EXTRA = {"26601", "2450", "30503", "6222", "17485", "87552", "60581", "79491", "30357"}
KNOB_EXTRA = {"4740", "3262", "14769", "2654"}
SKIP = {"1750", "70681", "2310", "35464", "6005", "77180", "77182", "13731", "76797", "32739",
        "72454", "2449", "67440", "10314"}                         # shapes the grid model can't express
ROT = {0: ((1, 0, 0), (0, 1, 0), (0, 0, 1)), 90: ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
       180: ((-1, 0, 0), (0, 1, 0), (0, 0, -1)), 270: ((0, 0, -1), (0, 1, 0), (1, 0, 0))}
DIRV = {"-z": (0, -1), "+z": (0, 1), "-x": (-1, 0), "+x": (1, 0)}
_BAD_WORDS = ("Clip", "Handle", "Tooth", "Socket", "Hinge", "Door Rail", "Petals", "Peghole", "Bar", "Pin",
              "Axle", "Minifig", "Projectile", "Wheel", "Tyre", "Spring", "Bracket", "Technic Beam",
              "Technic Bush", "Technic Angle", "Technic Cross", "Cylinder", "Wing", "Window", "Panel", "Arch",
              "Snapstud", "with Holes and", "Launcher", "Crown", "Ski", "Skate", "Corner", "without Corner",
              "Triangular", "Angled", "Cutout", "Inverted", "Dish", "Cone", "Rounded", "Half Round",
              "Open Stud", "Studs on", "Stud on", "Offset", "Groove with 1", "Curved Bottom", "Grille",
              "Ribbed", "Round", "Jumper", "Headlight", "with Pin", "Quadruple", "Double")


def geo_id(pid: str) -> str:
    """Inventory id -> LDraw geometry id (moved parts, printed parts use their base shape)."""
    if pid in MOVED:
        return MOVED[pid]
    return re.sub(r"(pr\d+|pat\d+(pr\d+)?)$", "", pid)


def part_name(g: str) -> str | None:
    t = tl.fetch(g + ".dat")
    return t.splitlines()[0][2:].strip() if t else None


def kind_of(name, w, d, h, nstuds):
    n = name
    if n is None:
        return None
    if n.startswith("~Moved to"):
        return "MOVED"
    if n.startswith(("Plate", "Brick", "Technic Brick")) and not any(b in n for b in _BAD_WORDS) and nstuds == w * d:
        return "fill"
    if n.startswith("Plate") and ("Offset" in n or "Groove with 1" in n or "with Two Studs" in n or "with 2 Studs" in n) \
            and nstuds >= 1 and "Corner" not in n:
        return "fill"                                   # jumpers: solid footprint, partial studs
    if n.startswith(("Plate", "Brick")) and "Round" in n and "Corner" not in n and "Half" not in n \
            and "Handle" not in n and "Holes" not in n and nstuds >= 1:
        return "fillint"                                # round footprint: hidden interior only
    if n.startswith("Brick") and ("Grille" in n or "Headlight" in n or "Studs on" in n or "Stud on" in n
                                  or "Ribbed" in n) and nstuds >= 1 and "Corner" not in n:
        return "fill"
    if n.startswith("Technic Brick") and nstuds == w * d and "Plate" not in n:
        return "fill"
    if n.startswith(("Tile", "=Tile")) and not any(b in n for b in ("Clip", "Handle", "Pin", "Triangular", "Angled", "Underside")):
        return "tile"
    if "Inverted" in n and n.startswith(("Slope", "=Slope")):
        return "islope"
    if n.startswith(("Slope", "=Slope")) or "Curved Top" in n:
        return "slope"
    if n.startswith("Brick  2 x  2 x  1 with Curved Bottom"):
        return "islope"
    if n.startswith(("Cone", "Plate  1 x  1 Round", "Brick  1 x  1 Round")):
        return "knob"
    return None


_CATALOG: dict = {}


def analyse(g: str) -> dict | None:
    """Grid description of one LDraw part: footprint w x d, height h (plates), kind, studs, slope directions."""
    if g in _CATALOG:
        return _CATALOG[g]
    nm = part_name(g)
    d = tl.part_info(g) if nm else None
    if not d:
        _CATALOG[g] = None
        return None
    polys, _ = tl.walk(g + ".dat")
    w = (d["x1"] - d["x0"]) / 20
    dd = (d["z1"] - d["z0"]) / 20
    h = (d["y1"] - d["y0"]) / 8
    reg = abs(w - round(w)) < 0.06 and abs(dd - round(dd)) < 0.06 and abs(h - round(h)) < 0.1 \
        and round(w) >= 1 and round(dd) >= 1 and round(h) >= 1
    out = dict(id=g, name=nm, x0=d["x0"], x1=d["x1"], y0=d["y0"], y1=d["y1"], z0=d["z0"], z1=d["z1"],
               studs=d["studs"], w=round(w), d=round(dd), h=round(h), reg=reg, polys=polys)
    out["kind"] = kind_of(nm, out["w"], out["d"], out["h"], len(d["studs"])) if reg else None
    if out["kind"] in ("slope", "islope"):                 # which sides drop away
        vs = [v for _, pp in polys for v in pp]

        def band(f):
            sel = [v for v in vs if f(v)]
            if not sel:
                return None
            return min(v[1] for v in sel) if out["kind"] == "slope" else max(v[1] for v in sel)
        zs, xs = d["z1"] - d["z0"], d["x1"] - d["x0"]
        B = {"-z": band(lambda v: v[2] < d["z0"] + 0.2 * zs), "+z": band(lambda v: v[2] > d["z1"] - 0.2 * zs),
             "-x": band(lambda v: v[0] < d["x0"] + 0.2 * xs), "+x": band(lambda v: v[0] > d["x1"] - 0.2 * xs)}
        desc = []
        for a, b in (("-z", "+z"), ("+z", "-z"), ("-x", "+x"), ("+x", "-x")):
            if B[a] is None or B[b] is None:
                continue
            if out["kind"] == "slope" and B[a] - B[b] > 3:
                desc.append(a)
            if out["kind"] == "islope" and B[b] - B[a] > 3:
                desc.append(a)
        out["desc"] = desc
    _CATALOG[g] = out
    return out


# ================================================================ part table + placement variants
class PartTable:
    """Every placeable inventory part, its grid footprint and all distinct rotated variants."""

    def __init__(self, inventory: collections.Counter, known_geometry_only: bool = True):
        """
        known_geometry_only: a printed part is only used if its base shape also appears in
        the inventory (the behaviour every Hammerhead version was generated with).
        Set False to resolve every printed part to its base shape.
        """
        self.PT, self.VAR = {}, {}
        ids = {p for p, _ in inventory}
        for (pid, _col) in inventory:
            if pid in self.PT or pid in SKIP:
                continue
            g = geo_id(pid)
            if known_geometry_only and g not in ids and pid not in MOVED:
                continue
            v = analyse(g)
            if not v:
                continue
            kind = v["kind"]
            if pid in CORNER:
                kind = "corner"
            elif pid in FILLINT_EXTRA:
                kind = "fillint"
            elif pid in KNOB_EXTRA:
                kind = "knob"
            elif pid in ("3660", "3665"):
                kind = "islope"
            if v["name"].startswith("Wing") or pid == "15624":
                kind = "wing"
            if kind not in ("fill", "fillint", "corner", "tile", "slope", "islope", "knob", "wing"):
                continue
            if not v["reg"] and kind != "wing":
                continue
            if kind == "wing":
                v = dict(v)
                v["w"] = math.ceil((v["x1"] - v["x0"]) / 20 - 0.05)
                v["d"] = math.ceil((v["z1"] - v["z0"]) / 20 - 0.05)
                v["h"] = max(1, round((v["y1"] - v["y0"]) / 8))
                if not v["studs"]:
                    continue
            if kind == "islope" and not v["studs"]:
                continue
            e = dict(pid=pid, geo=g, kind=kind, w=v["w"], d=v["d"], h=v["h"], x0=v["x0"], x1=v["x1"],
                     z0=v["z0"], z1=v["z1"], y1=v["y1"], studs=[tuple(s) for s in v["studs"]],
                     desc=v.get("desc") or [], name=v["name"])
            if kind == "corner":
                e["fp"] = [tuple(s) for s in v["studs"]]
            elif kind == "wing":
                e["fp"] = None
            else:
                e["fp"] = [(v["x0"] + 10 + 20 * a, v["z0"] + 10 + 20 * b) for a in range(v["w"]) for b in range(v["d"])]
            if kind == "tile" and "Corner Round" in v["name"]:
                # rounded corner = the bbox corner with no geometry vertex nearby
                vs = [p for _, t in v["polys"] for p in t]
                best = None
                for sx, cx in ((-1, v["x0"]), (1, v["x1"])):
                    for sz, cz in ((-1, v["z0"]), (1, v["z1"])):
                        dmin = min(math.hypot(p[0] - cx, p[2] - cz) for p in vs)
                        if best is None or dmin > best[0]:
                            best = (dmin, (sx, sz))
                e["round_corner"] = best[1]
            self.PT[pid] = e

    @staticmethod
    def rot_vec(rot, lx, lz):
        R = ROT[rot]
        return (R[0][0] * lx + R[0][2] * lz, R[2][0] * lx + R[2][2] * lz)

    def variants(self, pid):
        """[dict(rot, fp, studs, O, desc, rc, W, D)] - one entry per geometrically distinct rotation."""
        if pid in self.VAR:
            return self.VAR[pid]
        d = self.PT[pid]
        out, seen = [], set()
        rv = self.rot_vec
        if d["kind"] in ("corner", "slope", "islope", "wing") or "round_corner" in d:
            rots = (0, 90, 180, 270)
        else:
            rots = (0, 90) if d["w"] != d["d"] else (0,)
        for rot in rots:
            cs = [rv(rot, x, z) for x in (d["x0"], d["x1"]) for z in (d["z0"], d["z1"])]
            mx, mz = min(c[0] for c in cs), min(c[1] for c in cs)
            Mx, Mz = max(c[0] for c in cs), max(c[1] for c in cs)
            if d["studs"] and d["kind"] == "wing":
                s0 = rv(rot, *d["studs"][0])
                ox, oz = 10 - s0[0], 10 - s0[1]
                ox += 20 * math.ceil((-(mx + ox) - 0.6) / 20)
                oz += 20 * math.ceil((-(mz + oz) - 0.6) / 20)
                O = (round(ox, 3), round(oz, 3))
            else:
                O = (round(-mx / 10) * 10, round(-mz / 10) * 10)

            def cell(lx, lz, rot=rot, O=O):
                w = rv(rot, lx, lz)
                return (math.floor((w[0] + O[0]) / 20), math.floor((w[1] + O[1]) / 20))
            if d["fp"] is None:
                fp = sorted((a, b) for a in range(math.floor((mx + O[0] + 0.6) / 20), math.ceil((Mx + O[0] - 0.6) / 20))
                            for b in range(math.floor((mz + O[1] + 0.6) / 20), math.ceil((Mz + O[1] - 0.6) / 20)))
            else:
                fp = sorted({cell(*c) for c in d["fp"]})

            def on_grid(c, rot=rot, O=O):
                w = rv(rot, *c)
                return abs(((w[0] + O[0]) % 20) - 10) < 1.5 and abs(((w[1] + O[1]) % 20) - 10) < 1.5
            st = sorted({cell(*c) for c in d["studs"] if on_grid(c)})
            if min(a for a, b in fp) != 0 or min(b for a, b in fp) != 0:
                sh = (min(a for a, b in fp), min(b for a, b in fp))
                fp = [(a - sh[0], b - sh[1]) for a, b in fp]
                st = [(a - sh[0], b - sh[1]) for a, b in st]
                O = (O[0] - 20 * sh[0], O[1] - 20 * sh[1])
            desc = [tuple(round(a) for a in rv(rot, *DIRV[k])) for k in d["desc"]]
            rc = tuple(int(math.copysign(1, a)) for a in rv(rot, *d["round_corner"])) if "round_corner" in d else None
            key = (tuple(fp), tuple(st), tuple(sorted(desc)), rc)
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(rot=rot, fp=fp, studs=st, O=O, desc=desc, rc=rc,
                            W=max(a for a, b in fp) + 1, D=max(b for a, b in fp) + 1))
        self.VAR[pid] = out
        return out

    def ldpos(self, pid, var, i, j, L):
        """LDraw origin of a variant whose footprint's (0,0) cell sits at grid (i, j) on level L."""
        return (var["O"][0] + 20 * i, -8 * L - self.PT[pid]["y1"], var["O"][1] + 20 * j)


# ================================================================ shape: the ship as voxels
SHIP_DEFAULTS = dict(SHIFT=3, YC=216, BLADE_UP=216, BLADE_DN=216, BL_LEAD=0.0, BL_DEPTH=7, BL_SWEEP=5.0, BL_HALF=30,
                     BL_TIPHALF=10, HULL_R=55, HULL_J=(3, 41), DORS_R=30, DORS_J=(10, 29), NAC_R=40, NAC_J=(28, 48),
                     NAC_X=95, NAC_UP=40, NAC_DN=100, BRIDGE_J=(30, 36), EYE_R=36, EYE_X=50, EYE_J=5.0)
# parameter set of the v6 'round' base used for v6..v6g (later hand passes added nozzles, rings, etc.)
V6_PARAMS = dict(SHIP_DEFAULTS, SHIFT=2, LIFT=10, BLADE_UP=200, BLADE_DN=184, HULL_J=(3, 35), HULL_R=60, DORS_J=(9, 26),
                 DORS_R=34, NAC_J=(24, 40), NAC_R=46, NAC_X=120, NAC_UP=44, NAC_DN=108, BRIDGE_J=(26, 31), SEED=11,
                 BLADE_V2=1, BL_SWEEP=7, BL_ROOT=11, BL_TRAIL_SWEEP=1.5, BL_HALF=10, EYE_R=46, EYE_X=30, EYE_J=5.0,
                 STEP=False, STEP_NAC=False)
PRESETS = {"default": SHIP_DEFAULTS, "v6": V6_PARAMS}


def _qx(rel, dyrel, on=True):
    """Quantise |x| to 2-stud steps above the centreline (stepped hull) when `on`."""
    if not on or dyrel <= 0:
        return rel
    k = int(abs(rel) // 40)
    return 40 * k + 8


def build_shape(S: dict) -> dict:
    """Voxels {(i,j,L): region tag} for the hammerhead: blade, eye, hull, dorsal, bridge, nacelles, pylons, stand."""
    YC = S["YC"]
    V = {}

    def mark(k, tag, pri):
        if k not in V or V[k][1] < pri:
            V[k] = (tag, pri)
    for i in range(-14, 14):
        x = (i + 0.5) * 20
        for j in range(0, 52):
            z = (j + 0.5) * 20
            for L in range(0, 70):
                y = (L + 0.5) * 8
                dy = y - YC
                ax = abs(x)
                lim = S["BLADE_UP"] if dy > 0 else S["BLADE_DN"]
                Lq = (L // 3) * 3 + 1.5
                dyq = Lq * 8 - YC                       # blade profile quantised to brick courses
                if abs(dy) <= lim:
                    t = min(1.0, abs(dyq) / lim)
                    if S.get("BLADE_V2"):
                        lead = round(S["BL_SWEEP"] * t ** 1.15)
                        trail = round(S["BL_ROOT"] - S.get("BL_TRAIL_SWEEP", 1.5) * t)
                        half = S["BL_HALF"] - (S["BL_HALF"] - S.get("BL_TIPHALF", S["BL_HALF"])) * t
                        if lead <= j < trail and ax <= half:
                            mark((i, j, L), "blade", 3)
                    else:
                        lead = round(S["BL_LEAD"] + S["BL_SWEEP"] * t ** 1.4)
                        depth = round(S["BL_DEPTH"] - 2.5 * t)
                        half = S["BL_HALF"] - (S["BL_HALF"] - S["BL_TIPHALF"]) * (1 if t > 0.6 else 0)
                        if lead <= j < lead + depth and ax <= half:
                            mark((i, j, L), "blade", 3)
                if ax <= S["EYE_X"] and (dy / S["EYE_R"]) ** 2 + ((z - S["EYE_J"] * 20) / S["EYE_R"]) ** 2 <= 1:
                    mark((i, j, L), "eye", 5)
                j0, j1 = S["HULL_J"]
                if j0 <= j < j1:
                    s = 1.0 if j >= j0 + 3 else 0.75 + 0.25 * (j - j0) / 3
                    r = S["HULL_R"] * s
                    if (_qx(x, dy, S.get("STEP", True)) / r) ** 2 + (dy / r) ** 2 <= 1:
                        mark((i, j, L), "hull", 4)
                d0, d1 = S["DORS_J"]
                if d0 <= j < d1:
                    s = 1.0 if j >= d0 + 2 else 0.7 + 0.15 * (j - d0)
                    r = S["DORS_R"] * s
                    if (_qx(x, dy - S["HULL_R"], S.get("STEP", True)) / r) ** 2 + ((dy - S["HULL_R"]) / r) ** 2 <= 1:
                        mark((i, j, L), "dorsal", 4)
                b0, b1 = S["BRIDGE_J"]
                if b0 <= j < b1 and S["HULL_R"] - 10 <= dy <= S["HULL_R"] + 44 and ax <= 30 and \
                        ((z - (b0 + b1) / 2 * 20) / ((b1 - b0) * 10)) ** 2 + (x / 30) ** 2 <= 1.05:
                    mark((i, j, L), "bridge", 4)
                n0, n1 = S["NAC_J"]
                if n0 <= j < n1:
                    r = S["NAC_R"] + (4 if j >= n1 - 3 else 0)
                    if j < n0 + 2:
                        r = S["NAC_R"] * (0.75 + 0.125 * (j - n0))
                    for nm, cx, cy in (("nacL", -S["NAC_X"], S["NAC_UP"]), ("nacR", S["NAC_X"], S["NAC_UP"]),
                                       ("nacB", 0, -S["NAC_DN"])):
                        if (_qx(x - cx, dy - cy, S.get("STEP_NAC", S.get("STEP", True))) / r) ** 2 + ((dy - cy) / r) ** 2 <= 1:
                            mark((i, j, L), nm, 6)
                    if n0 + 3 <= j <= n1 - 6:
                        if abs(dy - S["NAC_UP"] + 8) <= 16 and ax <= S["NAC_X"]:
                            mark((i, j, L), "pylon", 2)
                        if ax <= 20 and -S["NAC_DN"] <= dy <= 0:
                            mark((i, j, L), "pylon", 2)
    out = {k: v[0] for k, v in V.items()}
    if S.get("STAND", True):
        SH, LIFT = S.get("SHIFT", 3), S.get("LIFT", 12)
        out = {(i, j, L + SH + LIFT): t for (i, j, L), t in out.items()}
        sj = S.get("STAND_J", (2, S["NAC_J"][1] - 3))
        for i in range(-4, 4):
            for j in range(sj[0], sj[1]):
                for L in range(0, SH):
                    out[(i, j, L)] = "stand"
        pillars = [(S.get("PILLAR_J", S["NAC_J"][0] + 6), -2, 4, 4.6), (S.get("FRONT_PILLAR_J", 5), -1, 2, 1.0)]
        for pj, i0, n, rr in pillars:
            for i in range(i0, i0 + n):
                for j in range(pj, pj + n):
                    if ((i - i0 - (n - 1) / 2) ** 2 + (j - pj - (n - 1) / 2) ** 2) > rr * n * n / 16 + 0.6:
                        continue
                    col = [L for (a, b, L), t in out.items() if a == i and b == j and t != "stand"]
                    if not col:
                        continue
                    for L in range(SH, min(col)):
                        out[(i, j, L)] = "stand"
    return out


def color_prefs(V: dict, S: dict) -> dict:
    """Preferred skin colours per voxel (Republic white/grey with red markings, dark engines)."""
    YC = S["YC"]
    P = {}
    SH = (S.get("SHIFT", 3) + S.get("LIFT", 12)) if S.get("STAND", True) else 0
    for (i, j, L), tag in V.items():
        y = (L - SH + 0.5) * 8
        dy = y - YC
        z = (j + 0.5) * 20
        if tag == "stand":
            P[(i, j, L)] = [0, 72]
            continue
        c = [15, 71]
        if tag == "hull":
            R = S["HULL_R"]
            if j < S["HULL_J"][0] + 3 or j in (11, 12, 24, 25):
                c = [4, 15]
            elif j in (18, 34):
                c = [72, 71]
            elif dy < -0.45 * R:
                c = [71, 72]
        elif tag == "dorsal":
            c = [4, 15] if j < S["DORS_J"][0] + 2 else ([72, 71] if j == 20 else [15, 71])
        elif tag == "blade":
            lim = S["BLADE_UP"] if dy > 0 else S["BLADE_DN"]
            t = min(1, abs(dy) / lim)
            if S.get("BLADE_V2"):
                lead = round(S["BL_SWEEP"] * t ** 1.15)
                depth = round(S["BL_ROOT"] - S.get("BL_TRAIL_SWEEP", 1.5) * t) - lead
            else:
                lead = S["BL_LEAD"] + S["BL_SWEEP"] * t ** 1.4
                depth = S["BL_DEPTH"] - 2.5 * t
            if z < (lead + 1.0) * 20:
                c = [4, 15]
            elif z > (lead + depth - 1.0) * 20:
                c = [72, 71]
            elif t > 0.72:
                c = [4, 15]
            elif 0.38 < t < 0.48:
                c = [72, 71]
        elif tag == "eye":
            rr = math.hypot(y - YC, z - S["EYE_J"] * 20)
            c = [72, 0] if rr > S["EYE_R"] - 14 else [0, 72]
        elif tag.startswith("nac"):
            n0, n1 = S["NAC_J"]
            if j >= n1 - 3:
                c = [0, 72]
            elif j < n0 + 2:
                c = [72, 71]
            elif j in (n0 + 4, n0 + 5, n1 - 7):
                c = [4, 15]
        elif tag == "pylon":
            c = [72, 71]
        P[(i, j, L)] = c
    return P


# ================================================================ builder
SKINPAL = [15, 71, 72, 4, 0]
RESERVE = {"6222", "60474", "92947", "3941", "17485"}     # held back for engines / stand
RESERVE_N = {"4032a": 12}


class Builder:
    """Greedy bottom-up brick placement over a voxel shape with live stud connectivity."""

    def __init__(self, V: dict, prefs: dict, pt: PartTable, inventory: collections.Counter):
        self.V, self.prefs, self.pt = V, prefs, pt
        self.INV = collections.Counter(inventory)
        self.P, self.OCC, self.STUD = [], {}, {}
        self.dsu = DisjointSet()
        self.allow_small = self.free_skin = False
        self.regions = None
        self.stand_top = 0
        self.int_pen = 7
        self.V0, self.crust, self.core = dict(V), set(), set()
        self.nodeco = set()
        self.SKIN = {k for k in V if any((k[0] + a, k[1] + b, k[2] + c) not in V
                                         for a, b, c in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)))}

    def colors(self, pid):
        return [c for (p, c), q in self.INV.items() if p == pid and q > 0]

    def add(self, pid, col, var, i, j, L, step=None, role="fill"):
        d = self.pt.PT[pid]
        assert self.INV[(pid, col)] > 0, (pid, col)
        self.INV[(pid, col)] -= 1
        idx = self.dsu.add()
        cells = [(i + a, j + b) for a, b in var["fp"]]
        for (a, b) in cells:
            for l in range(L, L + d["h"]):
                assert (a, b, l) not in self.OCC, (pid, a, b, l)
                self.OCC[(a, b, l)] = idx
        for (a, b) in cells:                                     # sits on studs below
            q = self.STUD.get((a, b, L - 1))
            if q is not None:
                self.dsu.union(q, idx)
        top = L + d["h"] - 1
        for (a, b) in [(i + x, j + y) for x, y in var["studs"]]:
            self.STUD[(a, b, top)] = idx
            q = self.OCC.get((a, b, top + 1))                     # under-hung part already above
            if q is not None and q != idx:
                self.dsu.union(q, idx)
        self.P.append(dict(pid=pid, color=col, var=var, i=i, j=j, L=L, h=d["h"], cells=cells,
                           step=L if step is None else step, role=role))
        return idx

    # ------------------------------------------------------------ structural fill
    def fill(self, levels, repair=False, phase=0, kinds=("fill", "fillint", "corner"), free_skin=False, regions=None):
        PT = self.pt.PT
        self.free_skin, self.regions = free_skin, regions
        cand = []
        for pid, d in PT.items():
            if d["kind"] not in kinds or (not free_skin and pid in RESERVE):
                continue
            for v in self.pt.variants(pid):
                cand.append((pid, v, len(v["fp"]) * d["h"]))
        cand.sort(key=lambda t: -t[2])
        demand = collections.Counter()
        for k in self.SKIN:
            if k not in self.OCC and k in self.prefs:
                demand[self.prefs[k][0]] += 1
        supply = collections.Counter()
        for (p, c), q in self.INV.items():
            if p in PT and PT[p]["kind"] in ("fill", "corner"):
                supply[c] += q * len(PT[p]["fp"]) * PT[p]["h"]
        for L in levels:
            m = (L + phase) % 4                               # alternate scan direction per layer
            KEY = [lambda c: (c[1], c[0]), lambda c: (-c[0], -c[1]), lambda c: (-c[1], -c[0]), lambda c: (c[0], c[1])][m]
            cells = [(i, j) for (i, j, l) in self.V if l == L and (i, j, L) not in self.OCC
                     and (regions is None or self.V[(i, j, l)] in regions)]
            cells.sort(key=lambda c: (0 if (c[0], c[1], L) in self.SKIN else 1,) + KEY(c))
            for (ai, aj) in cells:
                if (ai, aj, L) in self.OCC:
                    continue
                best = None
                for relaxed in (False, True):
                    for pid, v, vol in cand:
                        if best and 8 * vol + 400 < best[0] - 1000:
                            break
                        for (fa, fb) in v["fp"]:
                            r = self.evaluate(pid, v, vol, ai - fa, aj - fb, L, relaxed, repair, supply, demand)
                            if r and (best is None or r[0] > best[0]):
                                best = r
                    if best:
                        break
                if best:
                    score, pid, v, col, oi, oj, skinv = best
                    self.add(pid, col, v, oi, oj, L)
                    supply[col] -= len(v["fp"]) * PT[pid]["h"]
                    for k in skinv:
                        demand[self.prefs.get(k, [15])[0]] -= 1

    def evaluate(self, pid, v, vol, oi, oj, L, relaxed, repair, supply, demand):
        """Score one candidate placement (None = illegal). Rewards support, bridging, size; penalises orphans."""
        d = self.pt.PT[pid]
        h = d["h"]
        skinv, nsup, roots, below_shape, cross, sparts = [], 0, set(), False, 0, set()
        fps = [(oi + a, oj + b) for a, b in v["fp"]]
        for (ci, cj) in fps:
            for l in range(L, L + h):
                k = (ci, cj, l)
                if k not in self.V or k in self.OCC:
                    return None
                if self.regions is not None and self.V[k] not in self.regions:
                    return None
                if k in self.SKIN:
                    skinv.append(k)
        if self.free_skin:
            skinv = []
            if h > 1 and L < self.stand_top:
                return None
            if d["kind"] == "wing" and L >= self.stand_top:
                return None
        if skinv and not relaxed and d["name"].startswith("Technic Brick") and \
                not all(self.prefs.get(k, [15])[0] in getattr(self, "technic_ok", (0,)) for k in skinv):
            return None
        if skinv and d["kind"] == "fillint" and h > 1 and not getattr(self, "round_ok", False):
            return None
        if d["kind"] == "fillint" and skinv and not relaxed:
            return None
        if not skinv and d["kind"] != "fillint" and len(fps) < 3 and not self.allow_small:
            return None
        for (ci, cj) in fps:
            q = self.STUD.get((ci, cj, L - 1))
            if q is not None:
                nsup += 1
                roots.add(self.dsu.find(q))
                sparts.add(q)
                pb = self.P[q]["var"]
                if pb["W"] != pb["D"] and v["W"] != v["D"] and (pb["W"] > pb["D"]) != (v["W"] > v["D"]):
                    cross = 1
            if (ci, cj, L - 1) in self.V:
                below_shape = True
        if repair and nsup == 0:
            return None
        cols = self.colors(pid)
        if not cols:
            return None
        if not self.free_skin and pid in RESERVE_N and sum(self.INV[(pid, c)] for c in cols) <= RESERVE_N[pid]:
            return None
        if self.free_skin:
            col = next((c for c in (0, 72, 71, 15, 4) if c in cols), None)
            if col is None:
                col = max(cols, key=lambda c: self.INV[(pid, c)])
        elif skinv:
            allowed = None
            for k in skinv:
                pr = set(self.prefs.get(k, SKINPAL))
                allowed = pr if allowed is None else allowed & pr
            if relaxed:
                allowed = set(SKINPAL) | set(getattr(self, "extra_pal", ()))
            order = [c for c in self.prefs.get(skinv[0], SKINPAL) if c in allowed] + [c for c in SKINPAL if c in allowed]
            col = next((c for c in order if c in cols), None)
        else:
            nonskin = [c for c in cols if c not in SKINPAL]
            col = None
            if nonskin:
                col = max(nonskin, key=lambda c: self.INV[(pid, c)])
            else:
                sur = [c for c in cols if supply[c] - demand[c] > vol]
                if sur:
                    col = max(sur, key=lambda c: supply[c] - demand[c])
        if col is None:
            return None
        sup = nsup > 0 or not below_shape
        fset = set(fps)
        orphan = 0
        for (ci, cj) in fps:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (ci + di, cj + dj)
                if n in fset or (n[0], n[1], L) not in self.V or (n[0], n[1], L) in self.OCC:
                    continue
                if not any(((n[0] + ei, n[1] + ej) not in fset and (n[0] + ei, n[1] + ej, L) in self.V
                            and (n[0] + ei, n[1] + ej, L) not in self.OCC) for ei, ej in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    orphan += 1
        same = len(sparts) == 1 and sorted(self.P[next(iter(sparts))]["cells"]) == sorted(fps)
        brick_skin = h > 1 and skinv
        score = (1000 if sup else 0) + 60 * min(len(roots) - 1, 2) + 25 * min(len(sparts) - 1, 2) + 8 * vol \
            + 15 * cross - 40 * same - 45 * orphan + random.random() * 4
        score += -5 * vol if brick_skin else (3 * vol if h > 1 else 0)
        if not sup and L > 0 and not relaxed:
            score -= 200
        if skinv and not self.free_skin:
            score -= self.int_pen * (len(fps) * h - len(skinv))
        return (score, pid, v, col, oi, oj, skinv)

    # ------------------------------------------------------------ connectivity
    def components(self):
        return self.dsu.groups()

    def prune(self):
        """Drop everything outside the largest connected piece (parts go back to inventory)."""
        comp = self.components()
        if not comp:
            return 0
        keep = set(max(comp.values(), key=len))
        removed = [i for i in range(len(self.P)) if i not in keep]
        if not removed:
            return 0
        self.rebuild(set(removed))
        return len(removed)

    def rebuild(self, skip: set):
        """Re-add every part except those in `skip` (their inventory is returned)."""
        old = self.P
        self.P, self.OCC, self.STUD, self.dsu = [], {}, {}, DisjointSet()
        for p in old:
            self.INV[(p["pid"], p["color"])] += 1
        for idx, p in enumerate(old):
            if idx not in skip:
                self.add(p["pid"], p["color"], p["var"], p["i"], p["j"], p["L"], p["step"], p["role"])


def make(S: dict, inventory: collections.Counter, pt: PartTable | None = None, verbose=True) -> Builder:
    """Shape -> crust/core split -> region-ordered fill -> prune -> core repair -> stand from leftovers."""
    random.seed(S.get("SEED", 11))
    pt = pt or PartTable(inventory)
    V0 = build_shape(S)
    pr = color_prefs(V0, S)
    YC = S["YC"] + 8 * (S.get("SHIFT", 3) + S.get("LIFT", 12))
    cy = {"hull": YC, "dorsal": YC + S["HULL_R"], "nacL": YC + S["NAC_UP"], "nacR": YC + S["NAC_UP"],
          "nacB": YC - S["NAC_DN"], "bridge": YC, "blade": YC, "eye": YC, "pylon": YC}
    crust = set()                                   # thin top layer left for decoration
    if S.get("CRUST", 2):
        for (i, j, L), tag in V0.items():
            if tag == "stand" or (L + 0.5) * 8 <= cy[tag]:
                continue
            if any((i, j, L + u) not in V0 for u in range(1, S.get("CRUST", 2) + 1)) and tag not in ("eye", "pylon"):
                crust.add((i, j, L))
    V = {k: v for k, v in V0.items() if k not in crust}
    B = Builder(V, pr, pt, inventory)
    B.crust, B.V0, B.int_pen, B.S = crust, V0, S.get("INT_PEN", 7), S
    core = set()                                    # deep interior: filled last with leftovers
    for (i, j, L) in V:
        if all((i + a, j + b, L + c) in V for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-3, 0, 3)) and \
                all((i + a, j + b, L) in V for a, b in ((2, 0), (-2, 0), (0, 2), (0, -2))):
            core.add((i, j, L))
    B.core = core
    stand = {k: v for k, v in V.items() if v == "stand"}
    core -= set(stand)
    shell = {k: v for k, v in V.items() if k not in core and v != "stand"}
    maxL = max(k[2] for k in V)
    B.V = shell
    B.fill(range(0, maxL + 1), regions={"blade", "eye"})
    B.fill(range(0, maxL + 1), regions={"nacL", "nacR", "nacB", "pylon", "hull", "dorsal", "bridge"})
    B.fill(range(0, maxL + 1))
    rm, rm2 = B.prune(), 0
    for it in range(1, 5):
        B.fill(range(0, maxL + 1), phase=it)
        rm2 += B.prune()
    B.V = {k: v for k, v in V.items() if v != "stand"}
    B.allow_small = True
    B.fill(range(0, maxL + 1), repair=True)
    rm3 = B.prune()
    rm4 = 0
    if not S.get("NOSTAND_FILL"):
        B.V = stand
        SL = sorted({k[2] for k in stand})
        B.stand_top = S.get("SHIFT", 3)
        B.fill(SL, kinds=("fill", "fillint", "corner", "wing"), free_skin=True)
        rm4 = B.prune()
        B.fill(SL, kinds=("fill", "fillint", "corner", "wing"), free_skin=True, phase=1)
        rm4 += B.prune()
    B.V = V
    if verbose:
        print("voxels", len(V), "core", len(core), "shell", len(shell), "parts", len(B.P),
              "pruned", rm, rm2, rm3, rm4, "skin holes", len([k for k in B.SKIN if k not in B.OCC]))
    return B


# ================================================================ decoration passes
def exposed(B):
    return {(i, j, t) for (i, j, t) in B.STUD if (i, j, t + 1) not in B.OCC}


def deco_color(B, pid, under_cols, tag):
    cols = B.colors(pid)
    if not cols:
        return None
    if tag == "stand":
        return next((c for c in (0, 72, 71, 15, 4) if c in cols), None)
    pref = [40, 47] if (tag == "bridge" and B.pt.PT[pid]["kind"] == "slope") else []
    for c in pref + list(under_cols) + SKINPAL:
        if c in cols:
            return c
    if B.pt.PT[pid]["kind"] == "knob":
        return max(cols, key=lambda c: B.INV[(pid, c)])
    return None


def top_pass(B, pids, mode, maxn=10 ** 6, where=None, rng=None):
    """Place parts on exposed studs. mode: slope | ridge | corner | knob | tile."""
    PT = B.pt.PT
    n = 0
    for pid in pids:
        if pid not in PT:
            continue
        for v in B.pt.variants(pid):
            h = PT[pid]["h"]
            E = sorted(exposed(B), key=lambda k: (-k[2], k[1], k[0]))
            if rng:
                rng.shuffle(E)
            for (i, j, t) in E:
                if n >= maxn:
                    return n
                if (where and not where(i, j, t)) or (i, j, t) in B.nodeco:
                    continue
                fa, fb = v["fp"][0]
                oi, oj = i - fa, j - fb
                fps = [(oi + a, oj + b) for a, b in v["fp"]]
                if not all((c[0], c[1], t) in B.STUD and (c[0], c[1], t + 1) not in B.OCC
                           and all((c[0], c[1], l) not in B.OCC for l in range(t + 1, t + 1 + h)) for c in fps):
                    continue
                fset = set(fps)
                tg = B.V0.get((i, j, t), "")
                if mode == "slope":
                    if tg not in ("blade", "stand", "bridge") and any(dj != 0 for di, dj in v["desc"]):
                        continue
                    if not v["desc"]:
                        continue
                    good = True
                    for (di, dj) in v["desc"]:
                        for c in fps:                           # the low side must drop away
                            nb = (c[0] + di, c[1] + dj)
                            if nb in fset:
                                continue
                            if (nb[0], nb[1], t + 1) in B.OCC or ((nb[0], nb[1], t) in B.STUD and (nb[0], nb[1], t + 1) not in B.OCC):
                                good = False
                        for c in fps:                           # the high side must be backed
                            bk = (c[0] - di, c[1] - dj)
                            if bk in fset or (bk[0], bk[1], t + 1) in B.OCC or (bk[0], bk[1], t) in B.STUD:
                                continue
                            good = False
                    if not good:
                        continue
                elif mode == "ridge":
                    sides = ((1, 0), (-1, 0)) if v["D"] >= v["W"] else ((0, 1), (0, -1))
                    if not all(((c[0] + di, c[1] + dj) in fset) or not ((c[0] + di, c[1] + dj, t) in B.STUD
                                                                       and (c[0] + di, c[1] + dj, t + 1) not in B.OCC)
                               for c in fps for di, dj in sides):
                        continue
                elif mode == "corner":
                    si, sj = v["rc"]
                    cc = max(fps, key=lambda c: (c[0] * si + c[1] * sj))
                    if any(((cc[0] + a, cc[1] + b, t) in B.STUD and (cc[0] + a, cc[1] + b, t + 1) not in B.OCC)
                           or (cc[0] + a, cc[1] + b, t + 1) in B.OCC for a, b in ((si, 0), (0, sj))):
                        continue
                under = [B.P[B.STUD[(c[0], c[1], t)]]["color"] for c in fps]
                if tg == "stand" and mode == "slope" and any((c[0] + di, c[1] + dj, t) in B.V0 and (c[0] + di, c[1] + dj) not in fset
                                                             for c in fps for di, dj in v["desc"]):
                    continue
                if tg == "stand" and mode in ("knob", "ridge", "corner"):
                    continue
                col = deco_color(B, pid, under, tg)
                if col is None:
                    break
                B.add(pid, col, v, oi, oj, t + 1, role="deco")
                n += 1
    return n


def under_pass(B, pids):
    """Inverted slopes hung under the lowest fill parts where the shape drops away."""
    PT = B.pt.PT
    n = 0
    for pid in pids:
        if pid not in PT:
            continue
        h = PT[pid]["h"]
        for v in B.pt.variants(pid):
            bottoms = sorted({(c[0], c[1], p["L"]) for p in B.P if p["role"] == "fill" for c in p["cells"]},
                             key=lambda k: (k[2], k[1], k[0]))
            for (i, j, b) in bottoms:
                fa, fb = v["fp"][0]
                oi, oj = i - fa, j - fb
                fps = [(oi + a, oj + b2) for a, b2 in v["fp"]]
                if b - h < 0 or not all((c[0], c[1], b) in B.OCC and B.P[B.OCC[(c[0], c[1], b)]]["L"] == b
                                        and all((c[0], c[1], l) not in B.OCC for l in range(b - h, b)) for c in fps):
                    continue
                fset, good, wall = set(fps), True, 0
                for (di, dj) in v["desc"]:
                    for c in fps:
                        nb = (c[0] + di, c[1] + dj)
                        if nb in fset:
                            continue
                        if (nb[0], nb[1], b - 1) in B.OCC:
                            good = False
                        bk = (c[0] - di, c[1] - dj)
                        if bk not in fset and (bk[0], bk[1], b - 1) in B.OCC:
                            wall += 1
                if not good or (v["desc"] and wall == 0):
                    continue
                col = deco_color(B, pid, [B.P[B.OCC[(c[0], c[1], b)]]["color"] for c in fps], "")
                if col is None:
                    break
                B.add(pid, col, v, oi, oj, b - h, step=b, role="under")
                n += 1
    return n


def decorate(B) -> dict:
    rng = random.Random(5)
    PT = B.pt.PT
    out = dict(under=under_pass(B, ["3660", "3665", "32803", "13547"]))
    out["slope_big"] = top_pass(B, ["3039", "85984", "3040b", "28192", "3298", "4286", "15672", "3037", "60481",
                                    "61409", "66956", "37352"], "slope")
    out["slope_small"] = top_pass(B, ["54200", "54200pr0026", "85984pr0006", "3039pr0027", "3039pr0028"], "slope")
    out["ridge"] = top_pass(B, ["93273", "79756", "49307", "22388"], "ridge")
    out["knob"] = top_pass(B, ["4740", "3262", "98100", "59900", "4070", "3062b", "3062bpr0046", "6141", "85861"],
                           "knob", maxn=60, rng=rng)
    out["corner"] = top_pass(B, ["27925", "25269"], "corner")
    tiles = [p for p in sorted(PT, key=lambda p: -len(B.pt.variants(p)[0]["fp"]))
             if PT[p]["kind"] == "tile" and "round_corner" not in PT[p]]
    out["tile"] = top_pass(B, tiles, "tile")
    out["stand_tile"] = top_pass(B, ["27925", "25269"], "tile", where=lambda i, j, t: B.V0.get((i, j, t)) == "stand")
    return out


# ================================================================ output
REGION_NAMES = {"stand": "display stand", "blade": "hammerhead blade", "hull": "main hull", "dorsal": "dorsal spine",
                "nacL": "port engine", "nacR": "starboard engine", "nacB": "ventral engine", "pylon": "engine pylons",
                "eye": "blade hub", "bridge": "bridge tower", "": "detailing"}


def emit_ldr(B: Builder, path: str, title="Republic Hammerhead Corvette (custom MOC)", zoff=-480) -> tuple:
    """One STEP per plate layer, parts sorted fill-first then by position. Returns (parts, steps)."""
    steps = collections.defaultdict(list)
    for p in B.P:
        x, y, z = B.pt.ldpos(p["pid"], p["var"], p["i"], p["j"], p["L"])
        steps[p["step"]].append(dict(p, x=x, y=y, z=z + zoff, R=ROT[p["var"]["rot"]]))

    def fmt(v):
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        return "0" if s in ("-0", "") else s
    L = [f"0 {title}", f"0 Name: {os.path.basename(path)}", "0 Author: Claude (LDraw CAD Compiler)",
         "0 !LDRAW_ORG Unofficial_Model",
         "0 // Built ONLY from the supplied inventory. Colours are LDraw codes (Rebrickable IDs mapped where they differ).",
         "0 // Axes: bow = -Z, up = -Y, 1 stud = 20 LDU, 1 plate = 8 LDU. One STEP per plate layer, bottom to top.", ""]
    for n, s in enumerate(sorted(steps), 1):
        ps = sorted(steps[s], key=lambda p: (p["role"] != "fill", p["L"], p["z"], p["x"]))
        regs = collections.Counter(REGION_NAMES.get(
            collections.Counter(B.V0.get((c[0], c[1], p["L"]), "") for c in p["cells"]).most_common(1)[0][0], "detailing")
            for p in ps)
        L.append(f"0 // ===== STEP {n}: layer {s} ({', '.join(r for r, _ in regs.most_common(3))}) - {len(ps)} parts =====")
        for p in ps:
            g = B.pt.PT[p["pid"]]["geo"]
            if g != p["pid"] and p["pid"] not in MOVED:
                L.append(f"0 // printed part {p['pid']} shown with base geometry")
            m = " ".join(str(p["R"][r][c]) for r in range(3) for c in range(3))
            L.append(f"1 {ldcolor(p['color'])} {fmt(p['x'])} {fmt(p['y'])} {fmt(p['z'])} {m} {g}.dat")
        L.append("0 STEP")
        L.append("")
    open(path, "w").write("\n".join(L))
    return len(B.P), len(steps)


def inventory_report(B: Builder, original: collections.Counter) -> dict:
    """Usage vs. inventory: nothing may exceed what was supplied."""
    used = collections.Counter({k: original[k] - B.INV[k] for k in original if original[k] > B.INV[k]})
    over = [k for k in B.INV if B.INV[k] < 0]
    return dict(used=sum(used.values()), supplied=sum(original.values()), over_inventory=over)


# ================================================================ CLI
if __name__ == "__main__":
    import argparse
    import connectivity_check as cc
    ap = argparse.ArgumentParser(description="Generate a voxel-built Hammerhead corvette from an inventory CSV.")
    ap.add_argument("inventory_csv")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="v6")
    ap.add_argument("--params", help="JSON object of shape-parameter overrides")
    ap.add_argument("--no-deco", action="store_true")
    ap.add_argument("--out", default="hammerhead_generated.ldr")
    a = ap.parse_args()
    S = dict(PRESETS[a.preset])
    if a.params:
        S.update(json.loads(a.params))
    inv = load_inventory(a.inventory_csv)
    B = make(S, inv)
    if not a.no_deco:
        print("decoration:", decorate(B))
    n, steps = emit_ldr(B, a.out)
    print(f"{n} parts, {steps} steps -> {a.out}; inventory:", inventory_report(B, inv))
    met = cc.check(a.out)["metrics"]
    print({k: met[k] for k in ("parts", "overlaps", "components", "stud_links")})

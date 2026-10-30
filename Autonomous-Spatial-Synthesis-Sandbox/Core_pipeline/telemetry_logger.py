"""
core_pipeline/telemetry_logger.py
=================================
Geometry + measurement layer for the LDraw pipeline.

* LDraw library access (download + on-disk cache)
* sub-file walking with affine transforms
* per-part bounding boxes and top-stud positions  (part_info)
* .ldr loading and world-space bounding boxes       (load_ldr, world_bbox)
* intersection checking with tolerance metrics and the
  "legal overlap" rules (bar-in-clip, axle-in-hole, raised lips ...)
* a small TelemetryLogger that records every run as JSON lines

Conventions: LDraw units (1 stud = 20 LDU, 1 plate = 8 LDU, 1 brick = 24 LDU),
-Y is up. A placement matrix is stored row-major as 9 floats, exactly as it
appears on an LDraw type-1 line.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from collections import namedtuple
from dataclasses import dataclass, asdict, field
from functools import lru_cache

# ---------------------------------------------------------------- library access
LDRAW_BASE_URL = os.environ.get(
    "LDRAW_BASE_URL",
    "https://raw.githubusercontent.com/gkjohnson/ldraw-parts-library/master/complete/ldraw/")
CACHE_DIR = os.environ.get(
    "LDRAW_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ldraw_cache"))
_MISSING = "__MISSING__"


def fetch(name: str) -> str | None:
    """Return the text of an LDraw file (part, subpart or primitive), cached on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    name = name.lower().replace("\\", "/")
    cache_path = os.path.join(CACHE_DIR, name.replace("/", "__"))
    if os.path.exists(cache_path):
        text = open(cache_path, encoding="utf-8", errors="ignore").read()
        return None if text == _MISSING else text
    if name.startswith(("s/", "48/", "8/")):
        candidates = ["parts/" + name, "p/" + name]
    else:
        candidates = ["parts/" + name, "p/" + name, "parts/s/" + name]
    for c in candidates:
        try:
            text = urllib.request.urlopen(LDRAW_BASE_URL + c, timeout=20).read().decode("utf-8", "ignore")
            open(cache_path, "w").write(text)
            return text
        except Exception:
            pass
    open(cache_path, "w").write(_MISSING)
    return None


# ---------------------------------------------------------------- affine helpers
# An affine transform is (3x3 matrix, translation) or None for identity.
def apply(T, v):
    if T is None:
        return tuple(v)
    m, t = T
    return tuple(m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] + t[r] for r in range(3))


def compose(T, m2, t2):
    """T o (m2, t2)"""
    if T is None:
        return ([list(r) for r in m2], list(t2))
    m, t = T
    mm = [[sum(m[i][k] * m2[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return (mm, list(apply(T, tuple(t2))))


def mat9_apply(m9, pos, v):
    """Apply a row-major 9-float LDraw matrix + position to a local point."""
    x, y, z = pos
    return (m9[0] * v[0] + m9[1] * v[1] + m9[2] * v[2] + x,
            m9[3] * v[0] + m9[4] * v[1] + m9[5] * v[2] + y,
            m9[6] * v[0] + m9[7] * v[1] + m9[8] * v[2] + z)


def mat9_dir(m9, d):
    return (m9[0] * d[0] + m9[1] * d[1] + m9[2] * d[2],
            m9[3] * d[0] + m9[4] * d[1] + m9[5] * d[2],
            m9[6] * d[0] + m9[7] * d[1] + m9[8] * d[2])


# ---------------------------------------------------------------- part geometry
UNDERSIDE_STUDS = ("stud3", "stud4", "stud12", "stud16", "stud18", "stud22")
LEAF_STUDS = ("stud.", "stud2.", "stud2a.", "stud10", "stud15", "stud17", "stud6", "stud7", "stud9",
              "stud13", "stud20", "stud21", "stud23", "stud24", "stud25", "stud26", "stud27",
              "stud28", "stud29", "stud30")


def walk(name, T=None, polys=None, studs=None):
    """Recursively collect polygons and upward-facing top studs (x, z) of an LDraw file."""
    polys = [] if polys is None else polys
    studs = [] if studs is None else studs
    text = fetch(name)
    if text is None:
        return polys, studs
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "1" and len(p) >= 15:
            v = list(map(float, p[2:14]))
            sub = " ".join(p[14:]).lower()
            nt = compose(T, [v[3:6], v[6:9], v[9:12]], v[0:3])
            base = sub.replace("\\", "/").split("/")[-1]
            if base.startswith("stud") and not base.startswith(UNDERSIDE_STUDS):
                x, y, z = apply(nt, (0, 0, 0))
                if y <= 0.5 and nt[0][1][1] > 0.5:          # on the top face and pointing up
                    studs.append((x, z))
                if not base.startswith(LEAF_STUDS):
                    walk(sub, nt, polys, studs)
            elif base.startswith(UNDERSIDE_STUDS):
                walk(sub, nt, polys, [])                    # underside tubes: geometry only
            else:
                walk(sub, nt, polys, studs)
        elif p[0] in ("3", "4"):
            n = 3 if p[0] == "3" else 4
            polys.append((p[1], [apply(T, tuple(map(float, p[2 + 3 * i:5 + 3 * i]))) for i in range(n)]))
    return polys, studs


@lru_cache(maxsize=None)
def part_info(pid: str) -> dict | None:
    """Local bounding box (x0..z1) and top studs of a part, or None if unknown."""
    polys, studs = walk(pid + ".dat")
    if not polys:
        return None
    xs = [v[0] for _, p in polys for v in p]
    ys = [v[1] for _, p in polys for v in p]
    zs = [v[2] for _, p in polys for v in p]
    return dict(x0=min(xs), x1=max(xs), y0=min(ys), y1=max(ys), z0=min(zs), z1=max(zs),
                studs=sorted(set((round(a, 1), round(b, 1)) for a, b in studs)))


# ---------------------------------------------------------------- model loading
Placement = namedtuple("Placement", "pid color pos m9 line")


def load_ldr(path: str) -> list[Placement]:
    out = []
    for n, line in enumerate(open(path), 1):
        p = line.split()
        if len(p) == 15 and p[0] == "1":
            out.append(Placement(p[14][:-4], int(p[1]), tuple(map(float, p[2:5])),
                                 tuple(map(float, p[5:14])), n))
    return out


def count_steps(path: str) -> int:
    return sum(1 for ln in open(path) if ln.strip() == "0 STEP")


def world_bbox(pl: Placement):
    """World AABB [x0,y0,z0,x1,y1,z1] and world positions of the part's top studs."""
    v = part_info(pl.pid)
    if v is None:
        raise KeyError(f"unknown part {pl.pid} (line {pl.line})")
    cs = [mat9_apply(pl.m9, pl.pos, (a, b, c)) for a in (v["x0"], v["x1"])
          for b in (v["y0"], v["y1"]) for c in (v["z0"], v["z1"])]
    bb = [min(c[k] for c in cs) for k in range(3)] + [max(c[k] for c in cs) for k in range(3)]
    studs = [mat9_apply(pl.m9, pl.pos, (sx, v["y0"], sz)) for sx, sz in v["studs"]]
    return bb, studs


def bbox_from_local(pos, m9, lb):
    """AABB of a local box lb=(x0,y0,z0,x1,y1,z1) under a placement."""
    cs = [mat9_apply(m9, pos, (a, b, c)) for a in (lb[0], lb[3]) for b in (lb[1], lb[4]) for c in (lb[2], lb[5])]
    return [min(q[k] for q in cs) for k in range(3)] + [max(q[k] for q in cs) for k in range(3)]


def overlap_depths(A, B):
    """Per-axis overlap (positive = interpenetration) between two AABBs."""
    return [min(A[d + 3], B[d + 3]) - max(A[d], B[d]) for d in range(3)]


# ---------------------------------------------------------------- tolerances and rule sets
@dataclass
class Tolerances:
    overlap_eps: float = 1.0       # AABB interpenetration allowed on every axis before it counts
    lip_eps: float = 4.2           # raised stud rings / bars / handle lips allowed into the part above
    axle_clearance: float = 14.0   # min radial gap between an axle and round tiles around its tip
    tube_grid_eps: float = 0.05    # axle tip must sit on a plate's tube grid (fraction of a stud)
    stud_probe: float = 2.0        # how far along a stud's axis we probe for a receiving part
    probe_eps: float = 0.3         # slack on the receiving part's AABB


@dataclass
class Rules:
    """Part families that may legally intersect (real LEGO connections or hollow geometry)."""
    clips: set = field(default_factory=lambda: {"63868"})
    bars: set = field(default_factory=lambda: {"87994", "30374", "63965", "2714b", "15303"})
    axles: set = field(default_factory=lambda: {"32062", "3705", "59426", "32073", "24316", "61184"})
    holed: set = field(default_factory=lambda: {"32064a", "4032a", "60474", "3895", "3701", "72206"})
    lip: set = field(default_factory=lambda: {"60594", "92099", "41854", "48336"})
    tubed: set = field(default_factory=lambda: {"3795", "3020", "3022", "18980", "3666"})
    corner: set = field(default_factory=lambda: {"2357", "2420", "77844"})
    nested_pairs: list = field(default_factory=lambda: [{"27925", "4740"}, {"4345b", "4346"}])
    round_tiles: set = field(default_factory=lambda: {"98138", "27925"})
    # 1x2 clip plate: plate body and clip as two separate local volumes
    split_boxes: dict = field(default_factory=lambda: {
        "63868": [(-20, 0, -10, 20, 8, 10), (22.6, -5.4, -4, 34, 9.4, 4)]})


DEFAULT_TOL = Tolerances()
DEFAULT_RULES = Rules()


def sub_boxes(pl: Placement, bb, studs, rules: Rules = DEFAULT_RULES):
    """Finer collision volumes for non-box parts (L-shaped corners, clip plates)."""
    if pl.pid in rules.split_boxes:
        return [bbox_from_local(pl.pos, pl.m9, lb) for lb in rules.split_boxes[pl.pid]]
    if pl.pid in rules.corner:
        return [[sx - 10, bb[1], sz - 10, sx + 10, bb[4], sz + 10] for sx, _, sz in studs]
    return [bb]


def axle_in_tube(axle: Placement, plate: Placement, tol: Tolerances = DEFAULT_TOL) -> bool:
    """True if the axle's centre sits on the plate's under-tube grid (axle tips fit plate tubes)."""
    d = [axle.pos[i] - plate.pos[i] for i in range(3)]
    m = plate.m9
    lx = m[0] * d[0] + m[3] * d[1] + m[6] * d[2]
    lz = m[2] * d[0] + m[5] * d[1] + m[8] * d[2]
    return abs(lx / 20 - round(lx / 20)) < tol.tube_grid_eps and abs(lz / 20 - round(lz / 20)) < tol.tube_grid_eps


def is_exempt(a: Placement, b: Placement, rules: Rules = DEFAULT_RULES, tol: Tolerances = DEFAULT_TOL) -> bool:
    """Intersections that are real LEGO connections (they also count as joints for connectivity)."""
    ga, gb = a.pid, b.pid
    if (ga in rules.clips and gb in rules.bars) or (gb in rules.clips and ga in rules.bars):
        return True
    if (ga in rules.axles and gb in rules.holed) or (gb in rules.axles and ga in rules.holed):
        return True
    if any({ga, gb} == pair for pair in rules.nested_pairs):
        return True
    if (ga in rules.axles and gb == "4740") or (gb in rules.axles and ga == "4740"):
        return True                                  # axle tip in a dish's centre tube
    if (ga in rules.axles and gb in rules.tubed) or (gb in rules.axles and ga in rules.tubed):
        ax, pl = (a, b) if ga in rules.axles else (b, a)
        return axle_in_tube(ax, pl, tol)
    return False


def axle_gap_ok(a: Placement, b: Placement, rules: Rules = DEFAULT_RULES, tol: Tolerances = DEFAULT_TOL) -> bool:
    """Round tiles around (not on) an axle tip: check the true radial clearance."""
    if not ((a.pid in rules.axles and b.pid in rules.round_tiles) or
            (b.pid in rules.axles and a.pid in rules.round_tiles)):
        return False
    ax, t = (a, b) if a.pid in rules.axles else (b, a)
    return math.hypot(ax.pos[0] - t.pos[0], ax.pos[1] - t.pos[1]) >= tol.axle_clearance


def find_overlaps(parts, world=None, rules: Rules = DEFAULT_RULES, tol: Tolerances = DEFAULT_TOL):
    """
    Return a list of real clashes: dicts with both parts, their line numbers and the
    smallest per-axis interpenetration depth (LDU) - the tolerance metric for each clash.
    """
    world = world or [world_bbox(p) for p in parts]
    clashes = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            A, B = world[i][0], world[j][0]
            a, b = parts[i], parts[j]
            coarse = all(d > tol.overlap_eps for d in overlap_depths(A, B))
            if not (coarse or a.pid in rules.split_boxes or b.pid in rules.split_boxes):
                continue
            if is_exempt(a, b, rules, tol) or axle_gap_ok(a, b, rules, tol):
                continue
            if (a.pid in rules.lip or b.pid in rules.lip) and overlap_depths(A, B)[1] <= tol.lip_eps:
                continue
            worst = None
            for x in sub_boxes(a, *world[i], rules):
                for y in sub_boxes(b, *world[j], rules):
                    d = overlap_depths(x, y)
                    if all(v > tol.overlap_eps for v in d):
                        worst = max(worst or 0, min(d))
            if worst is not None:
                clashes.append(dict(a=a.pid, b=b.pid, line_a=a.line, line_b=b.line,
                                    pos_a=[round(v) for v in a.pos], pos_b=[round(v) for v in b.pos],
                                    depth=round(worst, 2)))
    return clashes


# ---------------------------------------------------------------- telemetry
class TelemetryLogger:
    """Collects metrics for one run and appends them as a JSON line to a log file."""

    def __init__(self, log_path: str | None = None, run_name: str = ""):
        self.log_path = log_path or os.environ.get("PIPELINE_LOG", "pipeline_telemetry.jsonl")
        self.record = dict(run=run_name, started=time.strftime("%Y-%m-%dT%H:%M:%S"), metrics={}, timings={})
        self._t = {}

    def start(self, key):
        self._t[key] = time.perf_counter()

    def stop(self, key):
        self.record["timings"][key] = round(time.perf_counter() - self._t.pop(key), 3)

    def metric(self, key, value):
        self.record["metrics"][key] = value

    def flush(self):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(self.record) + "\n")
        return self.record

    def summary(self) -> str:
        m = self.record["metrics"]
        return ", ".join(f"{k}={v}" for k, v in m.items() if not isinstance(v, (list, dict)))


def geometry_report(ldr_path: str, rules: Rules = DEFAULT_RULES, tol: Tolerances = DEFAULT_TOL,
                    logger: TelemetryLogger | None = None) -> dict:
    """Bounding-box / intersection metrics for a model."""
    logger = logger or TelemetryLogger(run_name=os.path.basename(ldr_path))
    logger.start("geometry")
    parts = load_ldr(ldr_path)
    world = [world_bbox(p) for p in parts]
    clashes = find_overlaps(parts, world, rules, tol)
    allb = [w[0] for w in world]
    extent = [round(min(b[k] for b in allb), 1) for k in range(3)] + [round(max(b[k + 3] for b in allb), 1) for k in range(3)]
    logger.stop("geometry")
    logger.metric("parts", len(parts))
    logger.metric("steps", count_steps(ldr_path))
    logger.metric("overlaps", len(clashes))
    logger.metric("max_overlap_depth", max((c["depth"] for c in clashes), default=0))
    logger.metric("extent_ldu", extent)
    logger.metric("size_studs", [round((extent[3] - extent[0]) / 20, 1), round((extent[5] - extent[2]) / 20, 1)])
    logger.metric("tolerances", asdict(tol))
    logger.metric("clashes", clashes[:25])
    return logger.record


if __name__ == "__main__":
    import sys
    rec = geometry_report(sys.argv[1])
    print(json.dumps({k: v for k, v in rec["metrics"].items() if k != "tolerances"}, indent=1))

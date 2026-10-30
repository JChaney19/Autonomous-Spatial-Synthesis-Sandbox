"""
core_pipeline/procedural_engine.py
==================================
Procedural model construction: matrices, inventory, placement and LDraw output.

* matrix helpers (RY rotations, axle/SNOT orientations, matrix x matrix, matrix x vector)
* Inventory      - Rebrickable CSV in, strict "never use more than you own" accounting
* Model          - step() / put() / raw() placement API and the .ldr writer
* macros         - reusable sub-assemblies (wheel + axle, clip + bar cannon, ...)
* build_shellback() - the AT-TC Shellback build, as a complete worked example

Units: LDraw (1 stud = 20 LDU, plate = 8, brick = 24), -Y is up, front of a vehicle is -X.
"""
from __future__ import annotations

import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import telemetry_logger as tl   # noqa: E402

# ---------------------------------------------------------------- colours / part aliases
# Rebrickable colour id -> LDraw colour code (only where they differ)
RB2LD = {41: 43, 45: 37, 182: 57, 326: 330, 1059: 365, 1095: 40, 1136: 402, 1103: 148}
LD2RB = {v: k for k, v in RB2LD.items()}
# Rebrickable ids whose LDraw geometry lives under another file name
GEO_ALIAS = {"2654": "2654a", "3660": "3660a", "3665": "3665a"}


def ldcolor(rb_color: int) -> int:
    return RB2LD.get(rb_color, rb_color)


def geo_id(pid: str) -> str:
    return GEO_ALIAS.get(pid, pid)


# ---------------------------------------------------------------- matrices (row-major 3x3 lists)
I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
AXLE_ALONG_Z = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]      # axle modelled along X turned to run across (Z)
STUDS_TO_REAR = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]      # studs-up part turned so its studs face +Z (SNOT)
BAR_FORWARD = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]        # bar along local +Y laid along -Z
BAR_FORWARD_REV = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]    # bar along local -Y (stop at origin) laid along -Z


def RY(deg: int):
    """Rotation about the vertical axis by 0/90/180/270 degrees."""
    return {0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 90: [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
            180: [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], 270: [[0, 0, -1], [0, 1, 0], [1, 0, 0]]}[deg % 360]


def mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mv(A, v):
    return [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]


def mirror_xz(side: int):
    """side=+1 identity, side=-1 half-turn: used to mirror an assembly to the other side."""
    return [[side, 0, 0], [0, 1, 0], [0, 0, side]]


def det(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


# ---------------------------------------------------------------- inventory
class Inventory:
    """Strict part budget loaded from a Rebrickable-style CSV (Part, Color/Color ID, Quantity)."""

    def __init__(self, csv_path: str):
        self.path = csv_path
        self.start = collections.Counter()
        self.meta = {}
        for r in csv.DictReader(open(csv_path)):
            col = int(r.get("Color ID") or r.get("Color"))
            self.start[(r["Part"], col)] += int(r["Quantity"])
            self.meta[(r["Part"], col)] = (r.get("Color", ""), r.get("Description", ""))
        self.left = collections.Counter(self.start)
        self.shortfalls = []

    def take(self, pid: str, col: int) -> bool:
        if self.left[(pid, col)] <= 0:
            self.shortfalls.append((pid, col))
            return False
        self.left[(pid, col)] -= 1
        return True

    def first_available(self, pid: str, colors) -> int | None:
        return next((c for c in colors if self.left[(pid, c)] > 0), None)

    def used(self) -> collections.Counter:
        return collections.Counter({k: self.start[k] - self.left[k] for k in self.start if self.start[k] > self.left[k]})

    def write_part_list(self, path: str):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Part", "Color ID", "Color", "Quantity", "Description"])
            for (p, c), n in sorted(self.used().items(), key=lambda kv: self.meta.get(kv[0], ("", ""))[1]):
                color, desc = self.meta.get((p, c), ("", ""))
                w.writerow([p, c, color, n, desc])

    def write_leftovers(self, path: str):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Part", "Color ID", "Color", "Quantity", "Description"])
            for (p, c), n in sorted(self.left.items()):
                if n > 0:
                    color, desc = self.meta.get((p, c), ("", ""))
                    w.writerow([p, c, color, n, desc])


# ---------------------------------------------------------------- model / placement API
class Model:
    """
    Collects placements grouped into named build steps and writes an LDraw file.
    put(): place by bounding box (centre in X/Z, bottom at ybot)  - for ordinary bricks/plates/slopes
    raw(): place by exact LDraw origin + matrix                    - for clips, bars, axles, hinges
    """

    def __init__(self, inventory: Inventory, title: str, notes: list[str] | None = None):
        self.inv = inventory
        self.title = title
        self.notes = notes or []
        self.steps = collections.OrderedDict()
        self._cur = ""

    def step(self, name: str):
        self._cur = name
        self.steps.setdefault(name, [])

    def _emit(self, col, pos, R, g):
        self.steps.setdefault(self._cur, []).append((ldcolor(col), pos, R, g))

    def put(self, pid: str, col: int, cx: float, cz: float, ybot: float, rot: int = 0, M=None):
        """Bounding box centred on (cx, cz), bottom face at ybot (remember: -Y is up)."""
        self.inv.take(pid, col)
        g = geo_id(pid)
        i = tl.part_info(g)
        if i is None:
            raise KeyError(f"no LDraw geometry for {pid}")
        R = M if M is not None else RY(rot)
        cs = [mv(R, [x, y, z]) for x in (i["x0"], i["x1"]) for y in (i["y0"], i["y1"]) for z in (i["z0"], i["z1"])]
        lo = [min(c[k] for c in cs) for k in range(3)]
        hi = [max(c[k] for c in cs) for k in range(3)]
        pos = (cx - (lo[0] + hi[0]) / 2, ybot - hi[1], cz - (lo[2] + hi[2]) / 2)
        self._emit(col, pos, R, g)
        return pos

    def raw(self, pid: str, col: int, x: float, y: float, z: float, R=I3):
        self.inv.take(pid, col)
        self._emit(col, (x, y, z), R, geo_id(pid))
        return (x, y, z)

    @property
    def count(self) -> int:
        return sum(len(r) for r in self.steps.values())

    @staticmethod
    def _fmt(v):
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        return "0" if s in ("-0", "") else s

    def write(self, path: str) -> int:
        L = [f"0 {self.title}", f"0 Name: {os.path.basename(path)}", "0 Author: Claude (LDraw CAD Compiler)",
             "0 !LDRAW_ORG Unofficial_Model"]
        L += [f"0 // {n}" for n in self.notes]
        L += ["0 // Axes: front = -X, up = -Y, 1 stud = 20 LDU, 1 plate = 8 LDU.", ""]
        for n, (name, rows) in enumerate(self.steps.items(), 1):
            if not rows:
                continue
            L.append(f"0 // ===== STEP {n}: {name} - {len(rows)} parts =====")
            for c, (x, y, z), R, g in rows:
                m = " ".join(self._fmt(R[a][b]) for a in range(3) for b in range(3))
                L.append(f"1 {c} {self._fmt(x)} {self._fmt(y)} {self._fmt(z)} {m} {g}.dat")
            L.append("0 STEP")
            L.append("")
        open(path, "w").write("\n".join(L))
        return self.count


# ---------------------------------------------------------------- macros (reusable sub-assemblies)
def wheel_on_axle(m: Model, x, side, axle_pid="3705", axle_col=0, axle_z=53, wheel_pid="72206",
                  wheel_col=1103, wheel_z=77, axle_ybot=-24):
    """Axle pushed through chassis holes across the vehicle, wheel on its outer end (side=+1/-1)."""
    m.put(axle_pid, axle_col, x, side * axle_z, axle_ybot, M=AXLE_ALONG_Z)
    m.put(wheel_pid, wheel_col, x, side * wheel_z, 0, rot=0 if side > 0 else 180)


def clip_cannon(m: Model, plate_pos, side, frame=RY(90), clip_col=72, bar_pid="2714b", bar_col=0,
                bar_offset=(30, 2, -16)):
    """
    1x2 clip plate (63868) + bar held in the clip, pointing forward.
    The assembly is defined in a 'ship' frame (bar along -Z) and rotated into the model by `frame`
    (RY(90) turns it so the barrel points along -X). side mirrors it for the other side.
    plate_pos = origin of the clip plate (plate centre, top surface).
    """
    Rs = mirror_xz(side)
    m.raw("63868", clip_col, *plate_pos, R=mm(frame, Rs))
    o = mv(frame, [side * bar_offset[0], bar_offset[1], bar_offset[2]])
    m.raw(bar_pid, bar_col, plate_pos[0] + o[0], plate_pos[1] + o[1], plate_pos[2] + o[2],
          R=mm(frame, BAR_FORWARD_REV if bar_pid in ("2714b", "63965") else BAR_FORWARD))


def snot_bracket_face(bracket_pos, rot):
    """World position of the first outward stud on a 99780-style 'up' bracket (face studs at z=-14, y=-2)."""
    return [bracket_pos[i] + v for i, v in enumerate(mv(RY(rot), [0, -2, -14]))]


# ---------------------------------------------------------------- worked example: AT-TC Shellback
def build_shellback(inv: Inventory) -> Model:
    m = Model(inv, "AT-TC Shellback - Republic troop carrier (custom MOC)",
              ["Built ONLY from unused_parts_after_v6e.csv (parts left over from Hammerhead v6e)."])
    Y_CH, Y_FL = -16, -40
    m.step("chassis - four black Technic 1x12 bricks (outer pair each side)")
    for z in (-50, -30, 30, 50):
        m.put("3895", 0, 0, z, Y_CH)
    m.step("axles and wheels - Technic axle 4 through both bricks on each side")
    for x in (-100, 100):
        for s in (1, -1):
            wheel_on_axle(m, x, s)
    m.step("floor - three black 4x4 round plates lock the two halves together")
    for x in (-80, 0, 80):
        m.put("60474", 0, x, 0, Y_FL)
    YF = Y_FL - 8
    m.step("lower walls - two courses of black Technic 1x12 bricks")
    for z in (-50, 50):
        m.put("3895", 0, 0, z, Y_FL)
        m.put("3895", 0, 0, z, Y_FL - 24)
    YW = Y_FL - 48
    m.step("front and rear bulkheads - white Technic 1x4, black inverted 60 slope, white rail plates")
    for xe, sg in ((-110, 1), (110, -1)):
        m.put("3701", 15, xe, 0, YF, rot=90)
        m.put("67440", 0, xe, 0, YF - 24, rot=0 if sg > 0 else 180)
    for z in (-20, 20):
        m.put("32028", 15, -110, z, YF - 96, rot=90)
        m.put("32028", 15, -110, z, YF - 104, rot=90)
        m.put("32028", 15, 110, z, YF - 96, rot=270)
        m.raw("48336", 0, 110, YF - 104 - 8, z, RY(270))
    m.step("interior - driver and co-driver seats, steering column, troop seats on the centre spine")
    for z in (-20, 20):
        m.put("1750", 72, -60, z, YF, rot=0)
    m.put("3829c01", 71, -90, 0, YF, rot=270)
    for x in (-20, 20, 60):
        m.put("1750", 72, x, 0, YF, rot=90 if x != 20 else 270)
    m.step("upper walls - red armour panels; the gap in the middle of each side is a troop door")
    for z in (-50, 50):
        for x in (-80, 80):
            m.put("60581", 4, x, z, YW)
    YL = YW - 72
    m.step("REMOVABLE TOP - two white 6x12 wings form a swept delta canopy (lifts straight off)")
    m.put("30355", 15, 0, -60, YL, rot=90)
    m.put("30356", 15, 0, 60, YL, rot=90)
    YC = YL - 8
    m.step("removable top - dorsal hump: white curved 10x2x2 slopes and a white tail fin")
    m.put("77182", 15, -20, -40, YC, rot=90)
    m.put("77180", 15, -20, 40, YC, rot=90)
    m.put("6239", 15, 40, 0, YC, rot=90)
    m.step("front plough - two black 3x8 wings clipped under the chassis, wide dark grey bumper")
    m.put("50305", 0, -120, -30, Y_CH + 8, rot=90)
    m.put("50304", 0, -120, 30, Y_CH + 8, rot=90)
    m.put("3007", 72, -160, 0, Y_CH, rot=90)
    m.step("nose - white triple wedge and clear headlights")
    m.put("48933", 15, -180, 0, Y_CH - 24, rot=90)
    for z in (-70, -50, 50, 70):
        m.put("54200", 47, -150, z, Y_CH - 24, rot=90)
    m.step("rear step - black 2x6 wing skids under the chassis, dark grey rear bumper, tail lights")
    for z in (-30, 30):
        m.put("78443", 0, 120, z, Y_CH + 8, rot=270)
    m.put("3007", 72, 160, 0, Y_CH, rot=90)
    for z in (-70, -50, 50, 70):
        m.put("98138", 36, 170, z, Y_CH - 24)
    m.step("removable top - wingtip gun pods with twin cannons, antenna")
    for side in (1, -1):
        m.put("3941", 71, 100, side * 100, YC)
        clip_cannon(m, (90, YC - 24 - 8, side * 100), side)
    m.put("3957b", 0, 110, -50, YC)
    return m


if __name__ == "__main__":
    import argparse
    import connectivity_check as cc
    ap = argparse.ArgumentParser(description="Build the example model, then check it.")
    ap.add_argument("inventory_csv")
    ap.add_argument("--out", default="at-tc_shellback.ldr")
    a = ap.parse_args()
    inv = Inventory(a.inventory_csv)
    model = build_shellback(inv)
    n = model.write(a.out)
    inv.write_part_list(os.path.splitext(a.out)[0] + "_partlist.csv")
    print(f"{n} parts written to {a.out}; shortfalls: {inv.shortfalls or 'none'}")
    met = cc.check(a.out, removable_keyword="removable top")["metrics"]
    print({k: met[k] for k in ("parts", "overlaps", "components", "stud_links", "joint_links")})
    print("removable top:", met["subassembly"])

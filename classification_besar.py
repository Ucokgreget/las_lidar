import laspy
import numpy as np

# ==============================
# PARAMETERS
# ==============================
INPUT_FILE = "20251211095257000.las"
OUTPUT_FILE = "classified_output_besar.las"

CHUNK_SIZE = 2_000_000      # turunkan kalau RAM kecil

GRID_SIZE = 2.0             # meter
GROUND_THRESHOLD = 0.3      # meter
LOW_VEG_MAX = 0.5           # class 3
MED_VEG_MAX = 2.0           # class 4 (di atas ini: kandidat 5/6)

ROOF_ROUGHNESS_MAX = 0.35   # naikkan -> lebih banyak dianggap building
MIN_POINTS_PER_CELL = 6     # turunkan kalau roof sering bolong

# ==============================
# HELPERS
# ==============================
def make_keys(x_bin: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    # pack (x_bin,y_bin) -> int64 key
    xb = x_bin.astype(np.int64, copy=False)
    yb = y_bin.astype(np.int64, copy=False) & 0xFFFFFFFF
    return (xb << 32) | yb

def group_min(keys: np.ndarray, values: np.ndarray):
    order = np.argsort(keys)
    k = keys[order]
    v = values[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    mins = np.minimum.reduceat(v, starts)
    gkeys = k[starts]
    return gkeys, mins

def group_sum_count_sumsq(keys: np.ndarray, values: np.ndarray):
    order = np.argsort(keys)
    k = keys[order]
    v = values[order].astype(np.float64, copy=False)
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    counts = np.add.reduceat(np.ones_like(v, dtype=np.int64), starts)
    sums = np.add.reduceat(v, starts)
    sumsq = np.add.reduceat(v * v, starts)
    gkeys = k[starts]
    return gkeys, counts, sums, sumsq

# ==============================
# PASS 0: READ HEADER (no big arrays)
# ==============================
with laspy.open(INPUT_FILE) as reader:
    header = reader.header
    total_points = header.point_count
    x0, y0 = header.mins[0], header.mins[1]

print("Reading LAS header...")
print("Total points:", total_points)

# ==============================
# PASS 1: MIN Z PER CELL
# ==============================
print("\nPASS 1/3: Computing min Z per grid cell...")
minz_dict = {}  # key(int64) -> min_z(float64)

with laspy.open(INPUT_FILE) as reader:
    for pts in reader.chunk_iterator(CHUNK_SIZE):
        x = np.asarray(pts.x)  # float64 chunk
        y = np.asarray(pts.y)
        z = np.asarray(pts.z)

        x_bin = np.floor((x - x0) / GRID_SIZE).astype(np.int32)
        y_bin = np.floor((y - y0) / GRID_SIZE).astype(np.int32)
        keys = make_keys(x_bin, y_bin)

        gk, mins = group_min(keys, z)

        # update global dict (loop over #cells in chunk, not #points)
        for k, m in zip(gk, mins):
            prev = minz_dict.get(k)
            if prev is None or m < prev:
                minz_dict[k] = float(m)

print("Cells found:", len(minz_dict))

# build sorted arrays for fast vectorized lookup via searchsorted
all_keys = np.fromiter(minz_dict.keys(), dtype=np.int64, count=len(minz_dict))
all_minz = np.fromiter(minz_dict.values(), dtype=np.float64, count=len(minz_dict))

sort_idx = np.argsort(all_keys)
all_keys = all_keys[sort_idx]
all_minz = all_minz[sort_idx]

# ==============================
# PASS 2: ROUGHNESS STATS (tall points only)
# ==============================
print("\nPASS 2/3: Computing roughness stats per cell (tall points)...")
count_dict = {}
sum_dict = {}
sumsq_dict = {}

with laspy.open(INPUT_FILE) as reader:
    has_return_number = ("return_number" in reader.header.point_format.dimension_names)

    for pts in reader.chunk_iterator(CHUNK_SIZE):
        x = np.asarray(pts.x)
        y = np.asarray(pts.y)
        z = np.asarray(pts.z)

        x_bin = np.floor((x - x0) / GRID_SIZE).astype(np.int32)
        y_bin = np.floor((y - y0) / GRID_SIZE).astype(np.int32)
        keys = make_keys(x_bin, y_bin)

        # vectorized local min lookup
        pos = np.searchsorted(all_keys, keys)
        # (keys should exist; if you want safety, you can check bounds here)
        local_min = all_minz[pos]
        height = z - local_min

        # tall mask
        tall = height > MED_VEG_MAX

        if not np.any(tall):
            continue

        kt = keys[tall]
        zt = z[tall]

        gk, gc, gs, gss = group_sum_count_sumsq(kt, zt)

        for k, c, s, ss in zip(gk, gc, gs, gss):
            k = int(k)
            count_dict[k] = count_dict.get(k, 0) + int(c)
            sum_dict[k] = sum_dict.get(k, 0.0) + float(s)
            sumsq_dict[k] = sumsq_dict.get(k, 0.0) + float(ss)

# build roughness arrays aligned with all_keys
counts = np.zeros_like(all_keys, dtype=np.int64)
sums = np.zeros_like(all_keys, dtype=np.float64)
sumsq = np.zeros_like(all_keys, dtype=np.float64)

# fill from dicts (loop over #cells)
key_to_index = {int(k): i for i, k in enumerate(all_keys)}
for k, c in count_dict.items():
    i = key_to_index.get(k)
    if i is not None:
        counts[i] = c
        sums[i] = sum_dict.get(k, 0.0)
        sumsq[i] = sumsq_dict.get(k, 0.0)

valid = counts >= MIN_POINTS_PER_CELL
mean = np.zeros_like(all_keys, dtype=np.float64)
var = np.zeros_like(all_keys, dtype=np.float64)
roughness = np.zeros_like(all_keys, dtype=np.float64)

mean[valid] = sums[valid] / counts[valid]
var[valid] = (sumsq[valid] / counts[valid]) - mean[valid] ** 2
roughness[valid] = np.sqrt(np.maximum(var[valid], 0.0))

# ==============================
# PASS 3: CLASSIFY + WRITE OUTPUT
# ==============================
print("\nPASS 3/3: Classifying and writing output...")
class_hist = np.zeros(256, dtype=np.int64)

with laspy.open(INPUT_FILE) as reader:
    out_header = reader.header.copy()
    with laspy.open(OUTPUT_FILE, mode="w", header=out_header) as writer:
        has_return_number = ("return_number" in reader.header.point_format.dimension_names)

        for pts in reader.chunk_iterator(CHUNK_SIZE):
            x = np.asarray(pts.x)
            y = np.asarray(pts.y)
            z = np.asarray(pts.z)

            num_ret = np.asarray(pts.number_of_returns)
            if has_return_number:
                ret_num = np.asarray(pts.return_number)
            else:
                ret_num = None

            x_bin = np.floor((x - x0) / GRID_SIZE).astype(np.int32)
            y_bin = np.floor((y - y0) / GRID_SIZE).astype(np.int32)
            keys = make_keys(x_bin, y_bin)

            pos = np.searchsorted(all_keys, keys)
            local_min = all_minz[pos]
            height = z - local_min

            cls = np.zeros(len(z), dtype=np.uint8)

            ground = height <= GROUND_THRESHOLD
            cls[ground] = 2

            non_ground = ~ground
            low_veg = non_ground & (height > GROUND_THRESHOLD) & (height <= LOW_VEG_MAX)
            med_veg = non_ground & (height > LOW_VEG_MAX) & (height <= MED_VEG_MAX)
            cls[low_veg] = 3
            cls[med_veg] = 4

            tall = non_ground & (height > MED_VEG_MAX)

            # roughness per point (via pos -> roughness array)
            r = roughness[pos]
            v = valid[pos]

            has_multi = num_ret > 1
            if ret_num is not None:
                not_last = ret_num < num_ret
            else:
                not_last = has_multi

            # base split
            veg_base = tall & v & (r > ROOF_ROUGHNESS_MAX)
            bld_base = tall & v & (r <= ROOF_ROUGHNESS_MAX)

            # veg boost (optional)
            veg = veg_base | (tall & v & (has_multi | not_last) & (r > (ROOF_ROUGHNESS_MAX * 0.8)))
            bld = bld_base

            cls[veg] = 5
            cls[bld] = 6

            # write back into point record and write
            pts.classification = cls
            writer.write_points(pts)

            # stats
            bh = np.bincount(cls, minlength=256)
            class_hist[:len(bh)] += bh

print("Done:", OUTPUT_FILE)

print("\nClass distribution:")
for c in [0, 2, 3, 4, 5, 6]:
    print(f"Class {c}: {int(class_hist[c])} points")

print("\nTuning cepat:")
print("- Building kurang: naikkan ROOF_ROUGHNESS_MAX (0.40–0.55) atau turunkan MIN_POINTS_PER_CELL (4–6)")
print("- Vegetasi ketarik jadi building: turunkan ROOF_ROUGHNESS_MAX (0.20–0.30)")
print("- Ground terlalu banyak: kecilkan GROUND_THRESHOLD (0.15–0.25) atau kecilkan GRID_SIZE (1.0)")
print("- Kalau masih MemoryError: turunkan CHUNK_SIZE (mis 500_000)")
import laspy
import numpy as np

# ==============================
# PARAMETERS
# ==============================
INPUT_FILE = "20251211095257000.las"
OUTPUT_FILE = "classified_output_besak.las"

GRID_SIZE = 2.0            # meter
GROUND_THRESHOLD = 0.3     # meter (<= ini dianggap ground dari local min)
HEIGHT_THRESHOLD = 2.0     # meter (objek tinggi -> kandidat building/veg)

# Roughness threshold (tuning)
ROOF_ROUGHNESS_MAX = 0.35  # makin kecil -> makin ketat roof harus rata
MIN_POINTS_PER_CELL = 10   # minimal point tall per cell supaya roughness tidak noisy

# ==============================
# LOAD DATA
# ==============================
print("Reading LAS file...")
las = laspy.read(INPUT_FILE)
n = las.header.point_count
print("Total points:", n)

# IMPORTANT: convert laspy views -> numpy arrays (fix recursion error)
x = np.asarray(las.x, dtype=np.float64)
y = np.asarray(las.y, dtype=np.float64)
z = np.asarray(las.z, dtype=np.float64)

num_ret = np.asarray(las.number_of_returns)
has_return_number = ("return_number" in las.point_format.dimension_names)
ret_num = np.asarray(las.return_number) if has_return_number else None

# ==============================
# STEP 1 — GRID BINNING
# (shift by min to keep bins smaller & stable)
# ==============================
x0 = x.min()
y0 = y.min()

x_bin = np.floor((x - x0) / GRID_SIZE).astype(np.int64)
y_bin = np.floor((y - y0) / GRID_SIZE).astype(np.int64)

# Build unique cell id via unique rows
cell_keys = np.stack([x_bin, y_bin], axis=1)
uniq_cells, inv = np.unique(cell_keys, axis=0, return_inverse=True)
n_cells = uniq_cells.shape[0]



print("Grid cells:", n_cells)

# ==============================
# STEP 2 — MIN Z PER CELL (vectorized)
# ==============================
print("Computing minimum elevation per grid...")
min_z = np.full(n_cells, np.inf, dtype=np.float64)
np.minimum.at(min_z, inv, z)

height = z - min_z[inv]

# ==============================
# STEP 3 — GROUND CLASSIFICATION
# ==============================
print("Classifying ground...")
ground_mask = height <= GROUND_THRESHOLD

las.classification[:] = 0
las.classification[ground_mask] = 2  # Ground

# ==============================
# STEP 4 — ROUGHNESS PER CELL (untuk titik tinggi)
# ==============================
print("Computing roughness for tall points...")
non_ground = ~ground_mask
tall = non_ground & (height > HEIGHT_THRESHOLD)

# bincount but only if there are tall points
counts = np.bincount(inv[tall], minlength=n_cells).astype(np.float64)

sumz = np.bincount(inv[tall], weights=z[tall], minlength=n_cells)
sumsq = np.bincount(inv[tall], weights=z[tall] ** 2, minlength=n_cells)

roughness = np.zeros(n_cells, dtype=np.float64)
valid = counts >= MIN_POINTS_PER_CELL
mean = np.zeros(n_cells, dtype=np.float64)
var = np.zeros(n_cells, dtype=np.float64)

mean[valid] = sumz[valid] / counts[valid]
var[valid] = (sumsq[valid] / counts[valid]) - (mean[valid] ** 2)
roughness[valid] = np.sqrt(np.maximum(var[valid], 0.0))

roughness_per_point = roughness[inv]

# ==============================
# STEP 5 — VEGETATION & BUILDING
# ==============================
print("Classifying vegetation and building...")

has_multi = num_ret > 1
if ret_num is not None:
    not_last = ret_num < num_ret
else:
    not_last = has_multi  # fallback

# Vegetation: tall + roughness besar + indikasi multiple/not-last return
vegetation_mask = (
    tall &
    (roughness_per_point > ROOF_ROUGHNESS_MAX) &
    (has_multi | not_last)
)

# Building: tall + roughness kecil (rata) + cell cukup padat
building_mask = (
    tall &
    (roughness_per_point <= ROOF_ROUGHNESS_MAX) &
    (counts[inv] >= MIN_POINTS_PER_CELL)
)

las.classification[vegetation_mask] = 5  # Vegetation
las.classification[building_mask] = 6    # Building

# ==============================
# SAVE OUTPUT
# ==============================
print("Saving output...")
las.write(OUTPUT_FILE)
print("Done:", OUTPUT_FILE)

# ==============================
# SUMMARY
# ==============================
cls = np.asarray(las.classification)
unique, counts_cls = np.unique(cls, return_counts=True)
print("\nClass distribution:")
for u, c in zip(unique, counts_cls):
    print(f"Class {int(u)}: {int(c)} points")

print("\nNotes:")
print("- Kalau building banyak ketukar jadi vegetasi: naikkan ROOF_ROUGHNESS_MAX (mis 0.45)")
print("- Kalau vegetasi banyak ketarik jadi building: turunkan ROOF_ROUGHNESS_MAX (mis 0.25)")
print("- Kalau hasil noisy: naikkan MIN_POINTS_PER_CELL atau kecilkan GRID_SIZE (mis 1.0)")
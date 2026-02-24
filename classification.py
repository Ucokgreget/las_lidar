import laspy
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ==============================
# PARAMETERS
# ==============================
INPUT_FILE = "20251212105617000.las"
OUTPUT_FILE = "classified_output.las"

GRID_SIZE = 2.0          # meter
GROUND_THRESHOLD = 0.3   # meter
HEIGHT_THRESHOLD = 2.0   # meter untuk building/vegetation

# ==============================
# LOAD DATA
# ==============================
print("Reading LAS file...")
las = laspy.read(INPUT_FILE)

print("Total points:", las.header.point_count)

x = las.x
y = las.y
z = las.z

# ==============================
# STEP 1 — GRID CREATION
# ==============================
print("Creating grid bins...")
x_bin = np.floor(x / GRID_SIZE)
y_bin = np.floor(y / GRID_SIZE)

# Mapping grid -> min z
print("Computing minimum elevation per grid...")
grid_min_z = defaultdict(lambda: np.inf)

for i in tqdm(range(len(z))):
    key = (x_bin[i], y_bin[i])
    if z[i] < grid_min_z[key]:
        grid_min_z[key] = z[i]

# ==============================
# STEP 2 — GROUND CLASSIFICATION
# ==============================
print("Classifying ground...")
ground_mask = np.zeros(len(z), dtype=bool)
height = np.zeros(len(z))

for i in tqdm(range(len(z))):
    key = (x_bin[i], y_bin[i])
    local_min = grid_min_z[key]
    height[i] = z[i] - local_min
    if height[i] <= GROUND_THRESHOLD:
        ground_mask[i] = True

las.classification[:] = 0
las.classification[ground_mask] = 2  # Ground class

# ==============================
# STEP 3 — VEGETATION & BUILDING
# ==============================
print("Classifying vegetation and building...")

non_ground = ~ground_mask

vegetation_mask = (
    non_ground &
    (height > HEIGHT_THRESHOLD) &
    (las.number_of_returns > 1)
)

building_mask = (
    non_ground &
    (height > HEIGHT_THRESHOLD) &
    (las.number_of_returns == 1)
)

las.classification[vegetation_mask] = 5   # Vegetation
las.classification[building_mask] = 6     # Building

# ==============================
# STEP 4 — SAVE OUTPUT
# ==============================
print("Saving output...")
las.write(OUTPUT_FILE)

print("Done.")

# ==============================
# SUMMARY
# ==============================
unique, counts = np.unique(las.classification, return_counts=True)
print("\nClass distribution:")
for u, c in zip(unique, counts):
    print(f"Class {u}: {c} points")
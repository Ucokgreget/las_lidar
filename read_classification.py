import numpy as np
import laspy

las = laspy.read("classified_output.las")

unique, counts = np.unique(las.classification, return_counts=True)

for u, c in zip(unique, counts):
    print(f"Class {u}: {c} points")
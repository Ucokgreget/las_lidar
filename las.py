import laspy
import numpy as np

las = laspy.read("20251212105617000.las")
height = las.z - np.min(las.z)
las.classification[:] = 0
print("Height stats:")
print("Mean:", np.mean(height))
print("Std:", np.std(height))
print("Percentile 5:", np.percentile(height, 5))
print("Percentile 10:", np.percentile(height, 10))
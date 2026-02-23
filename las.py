import laspy
import numpy as np
import matplotlib.pyplot as plt

las = laspy.read("20251212105617000.las")
for dim in las.point_format.dimensions:
    print(dim.name)
# print(las.x)
# print("------------------")
# print(las.y)
# print("------------------")
# print(las.z)
# print("------------------")
# print(las.return_number)
# print("------------------")
# print(las.gps_time)
# print("------------------")
# print(las.classification)

print(las.intensity.dtype)
print(np.min(las.intensity))
print(np.max(las.intensity))

red = las.red
green = las.green
blue = las.blue

coords = np.vstack((las.x, las.y, las.z)).T
first_point = coords[0]

distances = np.sqrt(np.sum((coords - first_point) ** 2, axis=1))
mask = distances < 500

filtered_points = las.points[mask]

# new_las = laspy.LasData(las.header)
# new_las.points = filtered_points.copy()
# new_las.write("hasil.las")


ground_points = las.points[
    las.number_of_returns == las.return_number
]


plt.hist(las.intensity, bins=50)
plt.title("Distribution of Intensity Values")
plt.savefig("intensity_histogram.png")


ground_mask = las.number_of_returns == las.return_number

plt.hist(las.intensity[ground_mask], bins=50, alpha=0.5, label="Ground")
plt.hist(las.intensity[~ground_mask], bins=50, alpha=0.5, label="Non-Ground")
plt.legend()
plt.savefig("compare.png")

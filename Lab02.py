import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Funkcja do dopasowania płaszczyzny za pomocą algorytmu RANSAC
def fit_plane_RANSAC(points, iterations=100, threshold=0.01):
    best_plane = None
    best_inliers = []
    best_error = np.inf

    for _ in range(iterations):
        random_indices = np.random.choice(points.shape[0], 3, replace=False)
        sample = points[random_indices]

        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]

        #  v1 i v2 mają prawidłową liczbę wymiarów (sprawdzam)
        if v1.ndim != 1 or v2.ndim != 1:
            continue

        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        distances = np.abs(np.dot(points - sample[0], normal))
        inliers = points[distances < threshold]

        error = np.mean(distances[distances < threshold])

        if len(inliers) > len(best_inliers) and error < best_error:
            best_plane = normal, sample[0]
            best_inliers = inliers
            best_error = error

    return best_plane


# Wczytanie chmur punktów z pliku CSV
def load_point_cloud(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            coords = line.strip().split()
            point = [float(coord) for coord in coords]
            points.append(point)
    return np.array(points)

def load_point_clouds(filenames):
    clouds = []
    for filename in filenames:
        cloud = load_point_cloud(filename)
        clouds.append(cloud)
    return clouds

def visualize_point_clouds(clouds, planes, clusters):
    fig = plt.figure(figsize=(15, 5))

    for i, (cloud, plane, cluster_labels) in enumerate(zip(clouds, planes, clusters), start=1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=cluster_labels)

        # Dodanie płaszczyzny
        point = plane[1]
        normal = plane[0]
        d = -point.dot(normal)
        xx, yy = np.meshgrid(np.linspace(min(cloud[:, 0]), max(cloud[:, 0]), 10),
                             np.linspace(min(cloud[:, 1]), max(cloud[:, 1]), 10))
        z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
        ax.plot_surface(xx, yy, z, alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Cloud {i}')

    plt.show()

clouds_xyz = load_point_clouds(['point_cloud_1.xyz', 'point_cloud_2.xyz', 'point_cloud_3.xyz'])



planes = [fit_plane_RANSAC(cloud) for cloud in clouds_xyz]

# algorytm KMeans do znalezienia trzech klastrów w chmurze punktów
kmeans_results = []
for cloud in clouds_xyz:
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(cloud)
    kmeans_results.append(labels)

visualize_point_clouds(clouds_xyz, planes, kmeans_results)

### Znalezienie rozłącznych chmur punktów za pomocą algorytmu k-średnich
###kmeans = KMeans(n_clusters=3)
###labels = kmeans.fit_predict(clouds_xyz)

# Dopasowanie płaszczyzny do każdej chmury punktów i określenie charakterystyki płaszczyzny
for i, cloud in enumerate(clouds_xyz):
    plane = fit_plane_RANSAC(cloud)
    normal = plane[0]

    print(f"Chmura punktów {i + 1}:")
    print("Wektor normalny do płaszczyzny:", normal)

    if np.abs(normal[2]) > np.abs(normal[0]) and np.abs(normal[2]) > np.abs(normal[1]):
        print("Płaszczyzna jest pozioma.")
    else:
        print("Płaszczyzna jest pionowa.")

    distances = np.abs(np.dot(cloud - plane[1], normal))
    mean_distance = np.mean(distances)
    print("Średnia odległość punktów do płaszczyzny:", mean_distance)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import open3d as o3d


# Znajdowanie rozłącznych chmur punktów za pomocą algorytmu DBSCAN
def find_clusters(clouds, eps=0.1, min_samples=3):
    clusters = []
    for cloud in clouds:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(cloud)
        clusters.append(labels)
    return clusters


# Wczytanie chmur punktów z pliku XYZ
def load_point_cloud(filename):
    return np.loadtxt(filename, skiprows=1)


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

        # Znajdź unikalne etykiety klastrów
        unique_labels = np.unique(cluster_labels)

        # Wyświetl chmurę punktów z odpowiednimi kolorami klastrów
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)
            ax.scatter(cloud[cluster_indices, 0], cloud[cluster_indices, 1], cloud[cluster_indices, 2],
                       c=[np.random.rand(3, ) for _ in cluster_indices[0]])

        # Dodanie płaszczyzny
        if plane is not None:
            a, b, c, d = plane
            # Wyliczenie punktów do rysowania płaszczyzny
            max_range = np.abs(np.max(cloud, axis=0) - np.min(cloud, axis=0)).max() * 2
            xx, yy = np.meshgrid(np.arange(-max_range, max_range, 0.2), np.arange(-max_range, max_range, 0.2))
            zz = (-a * xx - b * yy - d) / c
            ax.plot_surface(xx, yy, zz, alpha=0.5, color='cyan')

            # Sprawdzenie orientacji płaszczyzny
            orientation = "pionowa" if abs(c) < abs(a) and abs(c) < abs(b) else "pozioma"
            print(f"Orientacja płaszczyzny {i}: {orientation}")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Cloud {i}')

    plt.show()


clouds_xyz = load_point_clouds(['point_cloud_1.xyz', 'point_cloud_2.xyz', 'point_cloud_3.xyz'])

# Algorytm DBSCAN do znalezienia klastrów w chmurze punktów
dbscan_results = find_clusters(clouds_xyz)

# Dopasowanie płaszczyzny do każdej chmury punktów za pomocą open3d
planes = []
for cloud in clouds_xyz:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    #  ransac_plane z Open3D do dopasowania płaszczyzny
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=2000)

    planes.append(plane_model)

# Wyświetlenie chmur punktów oraz dopasowanych płaszczyzn
visualize_point_clouds(clouds_xyz, planes, dbscan_results)

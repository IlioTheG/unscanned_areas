import open3d as o3d


def material(transmission: object = 50000, alpha: float = 0.5) -> o3d.visualization.rendering.MaterialRecord:
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultLitTransparency'
    mat.base_color = [1, 1, 1, alpha]
    mat.base_roughness = 500
    mat.base_reflectance = 0
    mat.base_clearcoat = 0
    mat.thickness = 0
    mat.transmission = transmission
    mat.absorption_distance = 0
    mat.absorption_color = [0, 0, 0]
    mat.base_metallic = 0

    return mat


general_mat = material()
seen_mat = material(transmission=100000, alpha=.3)
unknown_mat = material(transmission=100000, alpha=.5)
occupied_mat = material(transmission=100000, alpha=.3)
scanned_mat = material(transmission=100000, alpha=1)
debug_mat = material(transmission=100000, alpha=1)

from importlib.metadata import version

import numpy as np
import pymeshlab as pml

PML_VER = version('pymeshlab')
# NOTE: assume the latest 2023.12 version

if PML_VER.startswith('0.2'):
    # monkey patch for 0.2 (only the used functions in this file!)
    pml.MeshSet.meshing_decimation_quadric_edge_collapse = pml.MeshSet.simplification_quadric_edge_collapse_decimation
    pml.MeshSet.meshing_isotropic_explicit_remeshing = pml.MeshSet.remeshing_isotropic_explicit_remeshing
    pml.MeshSet.meshing_remove_unreferenced_vertices = pml.MeshSet.remove_unreferenced_vertices
    pml.MeshSet.meshing_merge_close_vertices = pml.MeshSet.merge_close_vertices
    pml.MeshSet.meshing_remove_duplicate_faces = pml.MeshSet.remove_duplicate_faces
    pml.MeshSet.meshing_remove_null_faces = pml.MeshSet.remove_zero_area_faces
    pml.MeshSet.meshing_remove_connected_component_by_diameter = pml.MeshSet.remove_isolated_pieces_wrt_diameter
    pml.MeshSet.meshing_remove_connected_component_by_face_number = pml.MeshSet.remove_isolated_pieces_wrt_face_num
    pml.MeshSet.meshing_repair_non_manifold_edges = pml.MeshSet.repair_non_manifold_edges_by_removing_faces
    pml.MeshSet.meshing_repair_non_manifold_vertices = pml.MeshSet.repair_non_manifold_vertices_by_splitting
    pml.PercentageValue = pml.Percentage
    pml.PureValue = float
elif PML_VER.startswith('2022.2'):
    # monkey patch for 2022.2
    pml.PercentageValue = pml.Percentage
    pml.PureValue = pml.AbsoluteValue


def clean_mesh(
        verts, faces,
        v_pct=1, min_f=64, min_d=20, repair=True,
        remesh=True, remesh_size=0.01, remesh_iters=3, quads=False
):
    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(
            threshold=pml.PercentageValue(v_pct)
        )  # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.PercentageValue(min_d)
        )

    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
        ms.meshing_close_holes()

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(
            iterations=remesh_iters, targetlen=pml.PureValue(remesh_size), smoothflag=True
        )

    if quads:
        ms.meshing_tri_to_quad_dominant(level=1)

    ms.apply_coord_laplacian_smoothing(stepsmoothnum=3, boundary=True)

    # extract mesh
    m = ms.current_mesh()
    m.compact()
    new_verts = m.vertex_matrix()
    new_faces = m.face_matrix()

    print(f"[INFO] mesh cleaning: {_ori_vert_shape} --> {new_verts.shape}, {_ori_face_shape} --> {new_faces.shape}")

    return new_verts, new_faces


def decimate_mesh(
        verts, faces, target=5e4, backend="pymeshlab",
        remesh=False, optimalplacement=True
):
    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == "pyfqmr":
        import pyfqmr

        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        new_verts, new_faces, normals = solver.getMesh()
        new_vcolors = None
    else:
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.PercentageValue(1))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), optimalplacement=optimalplacement, preserveboundary=True, autoclean=True
        )

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(
                iterations=3, targetlen=pml.PercentageValue(1)
            )
        # extract mesh
        m = ms.current_mesh()
        m.compact()
        new_verts = m.vertex_matrix()
        new_faces = m.face_matrix()

    print(f"[INFO] mesh decimation: {_ori_vert_shape} --> {new_verts.shape}, {_ori_face_shape} --> {new_faces.shape}")

    return new_verts, new_faces


def laplacian_smooth(vertices, triangles, num_iterations=3, lambda_factor=0.5):
    for _ in range(num_iterations):
        # Compute vertex neighbors
        vertex_neighbors = {}
        for tri in triangles:
            for i in range(3):
                if tri[i] not in vertex_neighbors:
                    vertex_neighbors[tri[i]] = []
                for j in range(3):
                    if i != j:
                        vertex_neighbors[tri[i]].append(tri[j])

        # Update vertex positions
        new_vertices = vertices.copy()
        for i in range(vertices.shape[0]):
            if i in vertex_neighbors:
                neighbors = vertex_neighbors[i]
                neighbor_coords = vertices[neighbors]
                new_vertices[i] = (1 - lambda_factor) * vertices[i] + lambda_factor * np.mean(neighbor_coords,
                                                                                              axis=0)

        vertices = new_vertices

    return vertices

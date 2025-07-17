import os
import subprocess
import tempfile

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_cdt as cdt
from skimage.measure import marching_cubes
from skimage.measure import label as compute_cc
from skimage.filters import gaussian
from heapq import *
from numba import njit
import gzip
from scipy.ndimage import binary_dilation


import trimesh


from slam.io import write_mesh
from slam.differential_geometry import laplacian_mesh_smoothing

"""
Topology correction algorithm by Bazin et al.
Please cite the original papers if you use this code:
- Bazin et al. Topology correction using fast marching methods and its application to brain segmentation.
  MICCAI, 2005.
- Bazin et al. Topology correction of segmented medical images using a fast marching algorithm.
  Computer methods and programs in biomedicine, 2007.

The algorithm is re-implemented and accelerated using Python+Numba.

For the original Java implementation please see:
- https://github.com/piloubazin/cbstools-public/blob/master/de/mpg/cbs/core/shape/ShapeTopologyCorrection2.java
Or refer to the Nighres software:
- https://nighres.readthedocs.io/en/latest/shape/topology_correction.html
The look up table file "critical186LUT.raw.gz" is downloaded from Nighres:
- https://nighres.readthedocs.io/en/latest/
"""


class topology:
    """
    apply topology correction algorithm
    inpt: input volume
    threshold: used to create initial mask. We set threshold=16 for CortexODE.
    """

    def __init__(self, lut_file):
        bit, lut = tca_init_fill(
            lut_file, threshold=1.0
        )
        self.bit = bit
        self.lut = lut

    def apply(self, inpt, threshold=1.0):
        mask, init_pts = tca_mask_fill(inpt, threshold)
        output = tca_fill(inpt, mask, init_pts, self.bit, self.lut)
        return output  # , mask


@njit
def bit_map():
    """used for compute key"""
    twobit = np.array([2**k for k in range(26)], dtype=np.float64)
    bit = np.zeros(27, dtype=np.float64)
    bit[:13] = twobit[:13]
    bit[14:] = twobit[13:]
    bit = bit.copy().reshape(3, 3, 3)

    return bit


@njit
def check_topology(img, LUT, bit):
    """check the critical points"""
    res = False
    if img[1, 1, 1] == 1:  # inside the original object
        res = True
    else:  # check topology
        # load key from pattern: should keep dtypes as the same
        key = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    key += img[i, j, k] * bit[i, j, k]
        key = int(key)
        if LUT[key]:
            res = True
        else:
            res = False
        return res


""" tca_fill
This algorithm propagates from background to object.
It fills all holes and is used to fix WM segemntation. 
"""


def tca_mask_fill(levelset, threshold=1.0):
    """intialize processing mask"""

    initmask = np.zeros_like(levelset)
    initmask[2:-2, 2:-2, 2:-2] = 1
    initmask *= levelset <= threshold
    mask = binary_dilation(initmask, structure=np.ones([3, 3, 3]))
    init_pts = np.stack(np.where((mask - initmask) == 1)).astype(int).T

    return mask, init_pts


def tca_init_fill(path, threshold=1.0):
    """
    Initialization for topology correction.
    Step 1. load look up table
    Step 2. load bit map
    Step 3. compile the Numba by a toy example
    """
    # load look up tables
    with gzip.open(path, "rb") as lut_file:
        LUT = lut_file.read()

    # load bit map
    bit = bit_map()

    # create a toy example to compile the Numba
    img = (threshold - 0.1) * np.ones([10, 10, 10])
    img[4:6, 4:6, 4:6] = threshold + 0.1
    mask, init_pts = tca_mask_fill(img, threshold)
    img_fix = tca_fill(img, mask, init_pts, bit, LUT)

    return bit, LUT


@njit
def tca_fill(levelset, mask, init_pts, bit, LUT):
    """Configuration"""
    minDistance = 1e-5
    UNKNOWN = 10e10
    nx, ny, nz = levelset.shape
    # connectivity
    C6 = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    """Initialize indicators"""
    corrected = np.ones_like(levelset) * UNKNOWN  # gdm functions
    processed = np.zeros_like(levelset).astype(np.float64)
    inheap = np.zeros_like(levelset).astype(np.float64)
    mainval = 1e15
    maskval = -1e15
    maskval = np.max(mask * levelset)

    """add neighbors to the heap"""
    heap = []  # max heap
    for x0, y0, z0 in init_pts:
        processed[x0, y0, z0] = 1.0
        corrected[x0, y0, z0] = levelset[x0, y0, z0]
        if corrected[x0, y0, z0] < mainval:
            mainval = corrected[x0, y0, z0]

        for dx, dy, dz in C6:
            xn = x0 + dx
            yn = y0 + dy
            zn = z0 + dz
            if mask[xn, yn, zn] and not processed[xn, yn, zn]:
                heap.append((-levelset[xn, yn, zn], (xn, yn, zn)))
                inheap[xn, yn, zn] = 1.0
    heapify(heap)

    """Run Topology Correction"""
    while len(heap) > 0:
        # pop the heap
        val_, (x, y, z) = heappop(heap)
        val = -val_
        inheap[x, y, z] = 0.0

        if processed[x, y, z]:
            continue
        cube = processed[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2]
        non_critical = check_topology(cube, LUT, bit)

        if non_critical:
            # all correct: update and find new neighbors
            corrected[x, y, z] = val
            processed[x, y, z] = 1.0  #  update the current level
            mainval = val

            # find new neighbors
            for dx, dy, dz in C6:
                xn = x + dx
                yn = y + dy
                zn = z + dz
                if (
                    mask[xn, yn, zn]
                    and not processed[xn, yn, zn]
                    and not inheap[xn, yn, zn]
                ):
                    heappush(
                        heap,
                        (-min(levelset[xn, yn, zn], val - minDistance), (xn, yn, zn)),
                    )
                    inheap[xn, yn, zn] = True

    corrected += (mainval - corrected) * (1 - processed)
    corrected += (maskval - corrected) * (1 - mask)

    return corrected


""" tca_cut (to be validated)
This algorithm propagates from object to background.
It cuts all handles and is used to fix GM segemntation. 

Note: this function is not fully validated because we only use tca_fill for CortexODE.
"""


def tca_mask_cut(levelset, threshold=1.0):
    """intialize processing mask"""
    initmask = np.zeros_like(levelset)
    initmask[2:-2, 2:-2, 2:-2] = 1
    initmask *= levelset <= threshold
    mask = binary_dilation(initmask, structure=np.ones([3, 3, 3]))
    init_pts = np.stack(np.where(levelset == np.min(levelset))).astype(int).T
    return mask, init_pts


def tca_init_cut(path, threshold=1.0):
    """
    Initialization for topology correction.
    Step 1. load look up table
    Step 2. load bit map
    Step 3. compile the Numba by a toy example
    """
    # load look up tables
    with gzip.open(path, "rb") as lut_file:
        LUT = lut_file.read()

    # load bit map
    bit = bit_map()

    # create a toy example to compile the Numba
    img = (threshold + 0.1) * np.ones([10, 10, 10])
    img[4:6, 4:6, 4:6] = threshold - 0.1
    img[5, 5, 5] = threshold - 0.2
    mask, init_pts = tca_mask_cut(img, threshold)
    img_fix = tca_cut(img, mask, init_pts, bit, LUT)

    return bit, LUT


@njit
def tca_cut(levelset, mask, init_pts, bit, LUT):
    """Configuration"""
    minDistance = 1e-5
    UNKNOWN = 10e10
    nx, ny, nz = levelset.shape
    # connectivity
    C6 = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    """Initialize indicators"""
    corrected = np.ones_like(levelset) * UNKNOWN  # gdm functions
    processed = np.zeros_like(levelset).astype(np.float64)
    inheap = np.zeros_like(levelset).astype(np.float64)
    mainval = -1e15
    maskval = -1e15
    maskval = np.max(mask * levelset)

    """add neighbors to the heap"""
    heap = []  # max heap
    for x0, y0, z0 in init_pts:
        processed[x0, y0, z0] = 1.0
        corrected[x0, y0, z0] = levelset[x0, y0, z0]
        if corrected[x0, y0, z0] > mainval:
            mainval = corrected[x0, y0, z0]

        for dx, dy, dz in C6:
            xn = x0 + dx
            yn = y0 + dy
            zn = z0 + dz
            if mask[xn, yn, zn] and not processed[xn, yn, zn]:
                heap.append((levelset[xn, yn, zn], (xn, yn, zn)))
                inheap[xn, yn, zn] = 1.0
    heapify(heap)

    """Run Topology Correction"""
    while len(heap) > 0:
        # pop the heap
        val_, (x, y, z) = heappop(heap)
        val = val_
        inheap[x, y, z] = 0.0

        if processed[x, y, z]:
            continue
        cube = processed[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2]
        non_critical = check_topology(cube, LUT, bit)

        if non_critical:
            # all correct: update and find new neighbors
            corrected[x, y, z] = val
            processed[x, y, z] = 1.0  #  update the current level
            mainval = val

            # find new neighbors
            for dx, dy, dz in C6:
                xn = x + dx
                yn = y + dy
                zn = z + dz
                if (
                    mask[xn, yn, zn]
                    and not processed[xn, yn, zn]
                    and not inheap[xn, yn, zn]
                ):
                    heappush(
                        heap,
                        (max(levelset[xn, yn, zn], val - minDistance), (xn, yn, zn)),
                    )
                    inheap[xn, yn, zn] = True

    corrected += (mainval - corrected) * (1 - processed)
    corrected += (maskval - corrected) * (1 - mask)

    return corrected


def extract_hemi_mask_bounti(bounti_seg_file):
    """Generate a hemisphere white matter mask from dHCP BOUNTI tissue
    segmentation
    label 1 = Left hemisphere
    label 2 = Right hemisphere

    BOUNTI_NOMENCLATURE = {
    1: "eCSF_L",
    2: "eCSF_R",
    3: "Cortical_GM_L",
    4: "Cortical_GM_R",
    5: "Fetal_WM_L",
    6: "Fetal_WM_R",
    7: "Lateral_Ventricle_L",
    8: "Lateral_Ventricle_R",
    9: "Cavum_septum_pellucidum",
    10: "Brainstem",
    11: "Cerebellum_L",
    12: "Cerebellum_R",
    13: "Cerebellar_Vermis",
    14: "Basal_Ganglia_L",
    15: "Basal_Ganglia_R",
    16: "Thalamus_L",
    17: "Thalamus_R" ,
    18: "Third_Ventricle",
    19: "Fourth Ventricle",
    20: "Corpus_Callosum"
    }

    Parameters
    ----------
    bounti_seg: str
            Path of the whole brain tissue segmentation volume
    Returns
    wm_mask_vol: Nifti volume
    -------
    """
    bounti_seg = nib.load(bounti_seg_file)
    data = bounti_seg.get_fdata()
    new_data = data.copy()
    # insure integer values
    data = np.round(data)
    data = data.astype(np.uint16)

    # exclude non-brain tissues
    new_data[data == 1] = 0  # 1: "eCSF_L",
    new_data[data == 2] = 0  # 2: "eCSF_R",
    new_data[data == 10] = 0  # 10: "Brainstem",
    new_data[data == 11] = 0  # 11: "Cerebellum_L",
    new_data[data == 12] = 0  # 12: "Cerebellum_R",
    new_data[data == 13] = 0  # 13: "Cerebellar_Vermis",
    new_data[data == 18] = 0  # 18: "Third_Ventricle",
    new_data[data == 19] = 0  # 19: "Fourth Ventricle",

    # exclude cGM
    new_data[data == 3] = 0  # 3: "Cortical_GM_L",
    new_data[data == 4] = 0  # 4: "Cortical_GM_R",

    # exclude cavum septum
    new_data[data == 9] = 0  # 9: "Cavum_septum_pellucidum"

    # exclude corpus callosum
    new_data[data == 20] = 0  # 20: "Corpus_Callosum"

    # concatenate brain tissues within WM
    # label 1 = Left hemisphere
    # label 2 = Right hemisphere
    new_data[data == 5] = 1  # 5: "Fetal_WM_L",
    new_data[data == 6] = 2  # 6: "Fetal_WM_R",
    new_data[data == 7] = 1  # 7: "Lateral_Ventricle_L",
    new_data[data == 8] = 2  # 8: "Lateral_Ventricle_R",
    new_data[data == 14] = 1  # 14: "Basal_Ganglia_L",
    new_data[data == 15] = 2  # 15: "Basal_Ganglia_R",
    new_data[data == 16] = 1  # 16: "Thalamus_L",
    new_data[data == 17] = 2  # 17: "Thalamus_R" ,

    new_data = new_data.astype(np.uint16)
    wm_mask_vol = nib.Nifti1Image(new_data, affine=bounti_seg.affine)
    return wm_mask_vol


def seg2surf(lut_file, seg, sigma=0.5, alpha=16, level=0.55):
    """Extract a topologically spherical surface from a binary mask

    Parameters
    -----------
    seg: ndarray,
        binary volume to mesh
    sigma: float,
        standard deviation of gaussian blurring
    alpha: float,
        threshold for obtaining boundary of topology correction
    level: float,
        extracted surface level for Marching Cubes

    Returns
    -------
    mesh: trimesh.Trimesh,
        Raw topologically spherical triangular mesh

    Notes
    -----
    This function is a slightly modified version of the seg2surf function
    from the tca module of the CortexODE python package

    """

    # initialize topology correction
    topo_correct = topology(lut_file)
    # ------ connected components checking ------
    cc, nc = compute_cc(seg, connectivity=2, return_num=True)
    cc_id = 1 + np.argmax(
        np.array([np.count_nonzero(cc == i) for i in range(1, nc + 1)])
    )
    seg = (cc == cc_id).astype(np.float64)

    # ------ generate signed distance function ------
    sdf = -cdt(seg) + cdt(1 - seg)
    sdf = sdf.astype(float)
    sdf = gaussian(sdf, sigma=sigma)

    # ------ topology correction ------
    sdf_topo = topo_correct.apply(sdf, threshold=alpha)

    # ------ marching cubes ------
    v_mc, f_mc, _, _ = marching_cubes(
        -sdf_topo, level=-level, method="lewiner", allow_degenerate=False
    )
    # ------ generate mesh and perform minimal processing  -------
    # i.e. merging close vertices, degenerated faces and having consistent face
    # orientation
    mesh = trimesh.Trimesh(vertices=v_mc, faces=f_mc, process=True, validate=True)
    return mesh


def fix_mesh(path_mesh, path_mesh_fixed, path_container):
    """Improve quality of a triangular mesh using PyMesh fix_mesh.py script
    This function is a wrapper of the singularity execution of the pymesh
    fix_mesh.py script. This script will increase mesh quality by iteratively
        + correcting for obtuses triangles
        + Homogenise the surface of triangles (equilateral triangle)
        + Homogenise the length of edges
        + Homogenise the valence of vertices (closer to 6)
    This mesh fixing step is crucial for further processing of the mesh e.g.
    smoothing and registration

    Parameters
    ----------
    path_mesh: str
            Path of the raw triangular mesh to improve (should have .obj
            extension)
    path_mesh_fixed: str
            Path of the fixed triangular mesh (should have .obj
            extension)
    path_container: str
            Absolute path of the pymesh singularity container

    Returns
    -------
    status:
          Execution status of the mesh fixing process
    """

    print(f"Correcting {path_mesh}")
    cmd = [
        "singularity",
        "exec",
        "-B",
        os.path.dirname(path_mesh) + ":/data_in",
        "-B",
        os.path.dirname(path_mesh_fixed) + ":/data_out",
        path_container,
        "fix_mesh.py",
        "--detail",
        "high",
        os.path.join("/data_in", os.path.basename(path_mesh)),
        os.path.join("/data_out", os.path.basename(path_mesh_fixed)),
    ]
    status = subprocess.run(cmd)

    return status


def generate_mesh(
    lut_file,
    path_binary_mask,
    path_mesh,
    path_container,
    nb_smoothing_iter=10,
    smoothing_step=0.1,
):
    """Generate a topologically spherical and uniform triangular mesh from a
    binary mask volume

    Parameters
    ----------
    lut_file: str
            path of the critical LUT required by the topology correction algo
    path_binary_mask: str
            Path of a binary volume defining the object to mesh
    path_mesh: str
            Path of the triangular mesh generated from the volume
    path_container: str
            Path
    nb_smoothing_iter: int
    smoothing_step: float

    Returns
    -------

    """

    mask_volume = nib.load(path_binary_mask)
    mask = mask_volume.get_fdata()
    affine = mask_volume.affine
    mask = mask.astype(bool)
    # topologically correct raw triangular mesh
    mesh = seg2surf(lut_file, mask)
    # fix normals first

    with tempfile.NamedTemporaryFile(suffix="_raw.obj") as temp_raw:
        with tempfile.NamedTemporaryFile(suffix="_fixed.obj") as temp_fixed:
            # export mesh into .obj format
            mesh.export(temp_raw.name)
            print(mesh)
            fix_mesh(temp_raw.name, temp_fixed.name, path_container)
            # topologically correct and merely uniform triangular mesh
            fixed_mesh = trimesh.load(temp_fixed.name, force="mesh")
            print(fixed_mesh)
            # set the mesh into RAS+ scanner space
            # it eases visualization with FSLeyes or Anatomist
            smoothed_mesh = laplacian_mesh_smoothing(
                fixed_mesh, nb_smoothing_iter, smoothing_step, volume_preservation=True
            )
            smoothed_mesh.apply_transform(affine)
            write_mesh(smoothed_mesh, path_mesh)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a triangular mesh " "from a binary mask"
    )
    parser.add_argument("lut_file", help="path of the critical LUT file required for topology correction")
    parser.add_argument("path_mask", help="path of the input binary mask volume")
    parser.add_argument("path_mesh", help="path of the generated triangular " "mesh")
    parser.add_argument(
        "path_pymesh_container",
        help="path of the pymesh singularity " "container",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--nb_smoothing_iter",
        type=int,
        default=10,
        help="Number of smoothing iterations",
    )
    parser.add_argument(
        "-dt", "--delta", type=float, default=0.1, help="time delta used for smoothing"
    )
    args = parser.parse_args()
    generate_mesh(
        args.lut_file,
        args.path_mask,
        args.path_mesh,
        args.path_pymesh_container,
        args.nb_smoothing_iter,
        args.delta,
    )

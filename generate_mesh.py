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
import util.tca as utca

import trimesh

LUT_FILE = "./util/critical186LUT.raw.gz" # used in seg2surf()

def write_mesh(mesh, gifti_file):
    """Create a mesh object from two arrays

    fixme:  intent should be set !
    """
    coord = mesh.vertices
    triangles = mesh.faces
    carray = nib.gifti.GiftiDataArray(
        coord.astype(
            np.float32),
        "NIFTI_INTENT_POINTSET")
    tarray = nib.gifti.GiftiDataArray(
        triangles.astype(np.float32), "NIFTI_INTENT_TRIANGLE"
    )
    img = nib.gifti.GiftiImage(darrays=[carray, tarray])
    # , meta=mesh.metadata)

    nib.save(img, gifti_file)


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


def concatenate_labels_in_mask(seg_array, concatenated_labels):
    """Generate a hemisphere white matter the segmentation mask
    by concatenating the labels provided in concatenated_labels

    Parameters
    ----------
    seg_array: numpy array
            numpy array corresponding to the data from the nifti volume
             corresponding typically to whole brain tissue segmentation
    Returns
    binary_mask: numpy array
    -------
    """
    binary_mask = np.zeros_like(seg_array, dtype=np.uint16)
    # concatenate brain tissues within WM
    for label in concatenated_labels:
        binary_mask[seg_array == label] = 1

    return binary_mask

def seg2surf(seg, sigma=0.5, alpha=16, level=0.55):
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
    topo_correct = utca.topology(LUT_FILE)
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


def fix_mesh(path_mesh, path_mesh_fixed):
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
    path_container = "/scratch/gauzias/softs/pymesh_latest.sif"
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


def mesh_extraction(
    path_seg_vol,
    concatenated_labels,
    path_mesh,
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
    seg_vol_nifti = nib.load(path_seg_vol)
    mask = concatenate_labels_in_mask(seg_vol_nifti.get_fdata(), concatenated_labels)
    affine = seg_vol_nifti.affine
    mask = mask.astype(bool)
    print("mesh extraction")
    # topologically correct raw triangular mesh
    mesh = seg2surf(mask)
    # # set the mesh into RAS+ scanner space
    # # it eases visualization with FSLeyes or Anatomist
    #mesh.apply_transform(affine)

    with tempfile.NamedTemporaryFile(suffix="_raw.obj") as temp_raw:
        with tempfile.NamedTemporaryFile(suffix="_fixed.obj") as temp_fixed:
            # export mesh into .obj format
            mesh.export(temp_raw.name)
            print("mesh sampling refinment")
            fix_mesh(temp_raw.name, temp_fixed.name)
            # topologically correct and merely uniform triangular mesh
            fixed_mesh = trimesh.load(temp_fixed.name, force="mesh")
            print("mesh smoothing")
            smoothed_mesh = trimesh.smoothing.filter_laplacian(
                fixed_mesh,
                lamb=smoothing_step,
                iterations=nb_smoothing_iter,
                implicit_time_integration=False,
                volume_constraint=False
            )
            write_mesh(smoothed_mesh, path_mesh)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a triangular mesh " "from a binary mask"
    )
    parser.add_argument("-s", "--seg_vol", help="path of the input segmentation volume, *.nii[.gz]")
    parser.add_argument("-l", "--labels_concat", help="list of labels in the seg volume to concatenate in order to get the hemi mask, e.g. 2,34,26")

    parser.add_argument("-m", "--mesh_out", help="filename of the output triangular mesh, e.g. *.gii or *.stl")
    parser.add_argument(
        "-n",
        "--nb_smoothing_iter",
        type=int,
        default=20,
        help="Number of smoothing iterations",
    )
    parser.add_argument(
        "-dt", "--delta", type=float, default=0.3, help="time delta used for smoothing"
    )
    args = parser.parse_args()
    concatenated_labels = [int(item) for item in args.labels_concat.split(',')]
    print("labels from seg_vol to concatenate : ", concatenated_labels)
    mesh_extraction(
        args.seg_vol,
        concatenated_labels,
        args.mesh_out,
        args.nb_smoothing_iter,
        args.delta,
    )

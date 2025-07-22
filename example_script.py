import os

import tempfile
import numpy as np
import nibabel as nib
import util.generate_mesh as ugm

if __name__ == "__main__":
    TMP_PATH = ""
    path_pymesh_container = "/scratch/gauzias/softs/pymesh_latest.sif"
    lut_file = "/scratch/gauzias/code_gui/surface_processing/util/critical186LUT.raw.gz"
    nb_iter = 10
    dt = 0.1

    input_bounti_seg_mask = "/scratch/gauzias/data/datasets/MarsFet/output/svrtk_BOUNTI/output_BOUNTI_seg/haste/sub-0009/ses-0012/reo-SVR-output-brain-mask-brain_bounti-19.nii.gz"
    output_mesh_basename = "/scratch/gauzias/data/test_surface_processing/sub-0009_ses-0012_reo-SVR-output-brain-mask-brain_bounti-19.nii.gz"
    print(input_bounti_seg_mask)
    mask_filename = os.path.basename(input_bounti_seg_mask)
    # get WM hemi masks
    nii_white_hemi_mask = ugm.extract_hemi_mask_bounti(input_bounti_seg_mask)
    data_mask = nii_white_hemi_mask.get_fdata()
    for label,side in zip([1,2],['left','right']):
        # label 1 = Left hemisphere
        # label 2 = Right hemisphere
        output_mesh = output_mesh_basename+"."+side+".surf.gii"
        with tempfile.NamedTemporaryFile(suffix="."+side+".hemi_mask.nii.gz") as temp_mask:

            data_hemi = np.zeros_like(data_mask)
            data_hemi[data_mask==label] = 1
            nii_tmpfile_hemi = nib.Nifti1Image(data_hemi, nii_white_hemi_mask.affine)
            nib.save(nii_tmpfile_hemi, temp_mask)

            ugm.generate_mesh(
                lut_file,
                temp_mask.name,
                output_mesh,
                path_pymesh_container,
                nb_iter,
                dt,
            )

import os

import tempfile
import numpy as np
import nibabel as nib
import util.generate_mesh as ugm

if __name__ == "__main__":
    # parameters to be set by the user
    input_bounti_seg_mask = "/scratch/gauzias/data/datasets/MarsFet/output/svrtk_BOUNTI/output_BOUNTI_seg/haste/sub-0009/ses-0012/reo-SVR-output-brain-mask-brain_bounti-19.nii.gz"
    output_mesh = "/scratch/gauzias/data/test_surface_processing/sub-0009_ses-0012_reo-SVR-output-brain-mask-brain_bounti-19.left.white.gii"
    # list of labels to concatenate in order to get the white hemi mask
    concatenated_labels = [5,7,14,16]# for left
    #concatenated_labels = [6,8,15,17]# for right
    nb_smoothing_iter = 20
    smoothing_step = 0.3
    ugm.generate_mesh(
        input_bounti_seg_mask,
        concatenated_labels,
        output_mesh,
        nb_smoothing_iter,
        smoothing_step,
    )

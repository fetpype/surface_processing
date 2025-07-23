# surface_processing
Generation of fetal inner cortical (white) surfaces with spherical topology using 
a simple pipeline.
Note that this pipeline is similar to the non deep learning (DL) part of most 
DL tools such as DeepCSR or CortexODE. 
It consists of three steps:
1. Extraction of a closed binary mask of each hemisphere by combining the segmented structures from BOUNTI,
2. Correction of the topology in the voxel space using Bazin et al. tools (NiRes),
3. Extraction of the closed mesh using a classical marching cubes algorithm,
4. Refinement of the mesh (triangles, edges and vertices) using the docker of pymesh, enforcing a regular sampling of the surface,
5. Laplacian smoothing of the mesh using Trimesh.

# License
APACHE2.0 License inherited from Nighres

# Code provenance and references
util.tca correspond to the topology correction algorithm by Bazin et al., 
reimplemented in python+Numba by Qiang Ma in https://github.com/m-qiang/CortexODE.
Please cite the original papers if you use this code:
- Bazin et al. Topology correction using fast marching methods and its application to brain segmentation.
  MICCAI, 2005.
- Bazin et al. Topology correction of segmented medical images using a fast marching algorithm.
  Computer methods and programs in biomedicine, 2007.
- Q. Ma, L. Li, E. C. Robinson, B. Kainz, D. Rueckert and A. Alansary, "CortexODE: Learning Cortical Surface Reconstruction by Neural ODEs," in IEEE Transactions on Medical Imaging, vol. 42, no. 2, pp. 430-443, Feb. 2023, doi: 10.1109/TMI.2022.3206221.

# Dependencies
+ trimesh
+ nibabel
+ scikit-image
+ numba
+ pymesh (docker version)

# installation 

## Create the virtual anv and install the required packages
conda create --name surfaces python=3.8
conda activate surfaces
pip install -r requirements.txt

## Getting pymesh docker using singularity
```bash
singularity pull docker:pymesh/pymesh 
```
# Run the example script exmaple_script.py
Set the parameters and run the script




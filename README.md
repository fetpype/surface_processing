# surface_processing

APACHE2.0 License inherited from Nighres


util.tca correspond to the topology correction algorithm by Bazin et al.
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


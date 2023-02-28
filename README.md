
# CUDA LOD Generation

## About

This app generates a hierarchical LOD structure from your point cloud in CUDA. 


## Test Data

Do not share, only for internal use.

* [Saint Roman, 547M points](http://5.9.65.151/mschuetz/temporary/SaintRoman_cleaned_1094M_202003_halved.las)
* [CA21 Bunds, 975M points](http://5.9.65.151/mschuetz/temporary/CA21_bunds_9_to_14.las)

## Getting started

* Change path to point cloud in main_simlod.h
	* Look for ```// PICK POINT CLOUD TO PROCESS``` and pick the config you want
	* Make sure the config paths are correct. Defined in same file above.
* Check ```sampling_cuda_nonprogressive.h``` whether  ```#define MAX_BUFFER_SIZE``` fits your GPU and point cloud. 15GB works for 547M points. 30GB for 975M points? Note that this does not include the input data which requires another 16 * _numPoints_ bytes of GPU memory.
* Compile and run
* Hope it works. Check console that loading does work, prints message every 10M points.
* Try different splitting and voxelization methods in ```kernel.cu```
	* Change split: Go to  ```// PICK SPLIT METHOD``` and uncomment the desired one
	* Change voxelization: Go to ```// PICK VOXELIZATION METHOD``` and pick desired one

The kernels compute and print some stats, which affects performance. Stat printing can be deactivated in ```kernel.cu``` by setting ```PRINT_STATS``` to false. 

## Infos - Software Architecture

* The relevant code is in modules/simlod/sampling_cuda_nonprogressive
* CUDA code for LOD generation: ```kernel.cu```, ```countsorth.h.cu```. Most of it in ```countsorth.h.cu```. Recompilation at runtime is triggered by saving ```kernel.cu```
* CUDA code for rendering: ```render.cu```. Changes are immediately applied by saving this file. 
* Host-code is in ```sampling_cuda_nonprogressive.h```

## Infos - Algorithm

This app generates an LOD structure that is largely identical to Potree, but instead of points, lower LODs are filled with voxels. These voxels hold filtered color values that are representative of the voxels or points at higher levels of detail. The leaf nodes contain the original point data and no voxels.

* __kernel2__: Distribute points into an octree where each leaf node has at most 100k points
	* Create a 256³ (8 octree levels) counting grid and count the number of points in each cell
	* Recursively merge cells with <100k points.
	* Cells with > points are now leaf nodes of the octree. Some leaf nodes may be at level 8, while others may be at lower levels if counter cells were merged.
	* 256³ (8 levels) may not be sufficient to ensure that all leaf nodes have <100k points. All nodes that still have more than 100k point are then split again, but this time with a smaller counting grid, e.g. 16³ (4 levels). 
* __kernel3__: Populate currently empty lower levels of detail with voxels
	* Find all nodes with non-empty children (contain either voxels or points)
	* One node at a time, project points and voxels from their 8 direct child nodes into a voxel sampling grid. Each child point or voxel contributes its color values to all surrounding voxels of the sample grid, depending on their distance. 
	* We then iterate over the voxel sampling grid and extract the generated voxels from the voxel sampling grid. Voxels that do not contain a child point or voxel are ignored, even if child points or voxels contributed a color value due to their proximity. 

That's it. Rendering is done in ```render.cu```. The rendering is not yet optimized for speed. Each voxel/point is rendered to a single pixel. A screen-pass then iterates over all pixels, looks at surrounding pixels, and computes the color value based on the depth value, as well as weighted distances of each color. All of that is a quick hack and could probably be improved to get the right amount of sharpness vs. blurriness.
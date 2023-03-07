

namespace voxelize_singlecell_blockwise{

#include <cooperative_groups.h>
#include "lib.h.cu"
#include "methods_common.h.cu"

namespace cg = cooperative_groups;

void computeWorkload(Node* nodes, uint32_t numNodes, NodePtr* workload, uint32_t& workloadSize){

	auto grid = cg::this_grid();

	if(isFirstThread()){ 
		workloadSize = 0;
	}
	grid.sync();

	processRange(0, numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];
		
		int numSamplesInChildren = 0;
		bool allChildrenNonempty = true;
		for(int childIndex = 0; childIndex < 8; childIndex++){
			Node* child = node->children[childIndex];

			if(child){
				numSamplesInChildren += child->numPoints + child->numVoxels;
				if((child->numPoints + child->numVoxels) == 0){
					allChildrenNonempty = false;
				}
			}
		}
	
		bool isEmpty = node->numPoints == 0 && node->numVoxels == 0;

		if(isEmpty && allChildrenNonempty){
			uint32_t targetIndex = atomicAdd(&workloadSize, 1);
			workload[targetIndex] = node;
		}
	});

	grid.sync();

}

void main_voxelize(
	Allocator& allocator, 
	Box3 box, 
	int numPoints,
	void* nnnodes,
	uint32_t numNodes,
	void* sssorted)
{
	// Point* sorted = (Point*)sssorted;
	Node* nodes = (Node*)nnnodes;

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	NodePtr* workload              =  allocator.alloc<NodePtr*>(sizeof(Node) * numNodes, "workload");
	uint32_t& workloadSize         = *allocator.alloc<uint32_t*>(sizeof(uint32_t), "workload counter");
	uint32_t& totalVoxelBufferSize = *allocator.alloc<uint32_t*>(4, "total voxel buffer counter");
	uint32_t& nodeVoxelBufferSize  = *allocator.alloc<uint32_t*>(4, "node voxel buffer counter");
	uint32_t& clearCounter         = *allocator.alloc<uint32_t*>(4, "clear counter");

	if(isFirstThread()){
		totalVoxelBufferSize = 0;
		nodeVoxelBufferSize = 0;
		clearCounter = 0;
	}

	grid.sync();
	
	int gridSize               = VOXEL_GRID_SIZE;
	float fGridSize            = gridSize;
	int numCells               = gridSize * gridSize * gridSize;
	int acceptedCapacity       = 200'000;
	uint32_t& workIndexCounter = *allocator.alloc<uint32_t*>(4, "work index counter");
	uint64_t acceptedByteSize  = sizeof(uint32_t) * acceptedCapacity;
	uint32_t* accepteds           = allocator.alloc<uint32_t*>(grid.num_blocks() * acceptedByteSize, "list of accepted indices");
	uint32_t* accepted            = accepteds + grid.block_rank() * acceptedCapacity;

	// Create one voxelgrid per workgroup, and a <voxelGrid> pointer that points to the active workgroup's memory
	uint64_t voxelGridByteSize = 4 * sizeof(uint32_t) * numCells;
	uint32_t* voxelGrids       = allocator.alloc<uint32_t*>(grid.num_blocks() * voxelGridByteSize, "voxel sampling grids");
	uint32_t* voxelGrid        = voxelGrids + grid.block_rank() * 4 * numCells;

	uint64_t& globalAllocatorOffset = *allocator.alloc<uint64_t*>(8);

	__shared__ uint32_t sh_workIndex;
	__shared__ uint32_t sh_numAccepted;

	// initially clear all voxel grids
	clearBuffer(voxelGrids, 0, grid.num_blocks() * voxelGridByteSize, 0);

	grid.sync();

	if(isFirstThread()){
		globalAllocatorOffset = allocator.offset;

		printf("allocator.offset: ");
		printNumber(allocator.offset, 10);
		printf("\n");
	}
	grid.sync();

	// loop from bottom of hierarchy to top until all work done, 
	// but limit loop range to max octree depth to be safe
	for(int abc = 0; abc < 20; abc++){

		grid.sync();

		computeWorkload(nodes, numNodes, workload, workloadSize);
		if(grid.thread_rank() == 0){
			workIndexCounter = 0;
		}
		grid.sync();

		if(workloadSize == 0) break;

		while(workIndexCounter < workloadSize){

			block.sync();

			if(block.thread_rank() == 0){
				sh_workIndex = atomicAdd(&workIndexCounter, 1);
				sh_numAccepted = 0;
			}

			block.sync();

			if(sh_workIndex >= workloadSize) break;

			// retrieve the node that this block should process
			Node* node = workload[sh_workIndex];
			vec3 boxSize = node->max - node->min;
			vec3 childSize = boxSize * 0.5f;

			block.sync();

			for(int childIndex = 0; childIndex < 8; childIndex++){
				Node* child = node->children[childIndex];

				if(child == nullptr) continue;

				block.sync();

				// POINTS
				int numIterations = child->numPoints / block.num_threads() + 1;
				for(int it = 0; it < numIterations; it++){
					int pointIndex = block.num_threads() * it + block.thread_rank();

					if(pointIndex >= child->numPoints) continue;

					Point point = child->points[pointIndex];

					// project to node's 128³ sample grid
					int ix = clamp(fGridSize * (point.x - node->min.x) / boxSize.x, 0.0f, fGridSize - 1.0f);
					int iy = clamp(fGridSize * (point.y - node->min.y) / boxSize.y, 0.0f, fGridSize - 1.0f);
					int iz = clamp(fGridSize * (point.z - node->min.z) / boxSize.z, 0.0f, fGridSize - 1.0f);

					int voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

					uint8_t* rgba = (uint8_t*)&point.color;
					
					atomicAdd(&voxelGrid[4 * voxelIndex + 0], rgba[0]);
					atomicAdd(&voxelGrid[4 * voxelIndex + 1], rgba[1]);
					atomicAdd(&voxelGrid[4 * voxelIndex + 2], rgba[2]);
					uint32_t oldCount = atomicAdd(&voxelGrid[4 * voxelIndex + 3], 1);

					if(oldCount == 0){
						uint32_t acceptedIndex = atomicAdd(&sh_numAccepted, 1);
						accepted[acceptedIndex] = voxelIndex;
					}
				}

				block.sync();

				// VOXELS
				numIterations = child->numVoxels / block.num_threads() + 1;
				for(int it = 0; it < numIterations; it++){
					int pointIndex = block.num_threads() * it + block.thread_rank();

					if(pointIndex >= child->numVoxels) continue;

					Point point = child->voxels[pointIndex];

					// project to node's 128³ sample grid
					int ix = clamp(fGridSize * (point.x - node->min.x) / boxSize.x, 0.0f, fGridSize - 1.0f);
					int iy = clamp(fGridSize * (point.y - node->min.y) / boxSize.y, 0.0f, fGridSize - 1.0f);
					int iz = clamp(fGridSize * (point.z - node->min.z) / boxSize.z, 0.0f, fGridSize - 1.0f);

					int voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

					uint8_t* rgba = (uint8_t*)&point.color;
					
					atomicAdd(&voxelGrid[4 * voxelIndex + 0], rgba[0]);
					atomicAdd(&voxelGrid[4 * voxelIndex + 1], rgba[1]);
					atomicAdd(&voxelGrid[4 * voxelIndex + 2], rgba[2]);
					uint32_t oldCount = atomicAdd(&voxelGrid[4 * voxelIndex + 3], 1);

					if(oldCount == 0){
						uint32_t acceptedIndex = atomicAdd(&sh_numAccepted, 1);
						accepted[acceptedIndex] = voxelIndex;
					}
				}

				block.sync();
			}

			block.sync();

			// now allocate memory for the voxels of this node
			Point* voxelBuffer = nullptr;
			if(block.thread_rank() == 0){
				uint64_t bufferOffset = atomicAdd(&globalAllocatorOffset, 16ull * sh_numAccepted);
				voxelBuffer = reinterpret_cast<Point*>(allocator.buffer + bufferOffset);
				node->voxels = voxelBuffer;
				node->numVoxels = sh_numAccepted;
			}

			block.sync();

			// EXTRACT
			int numIterations = sh_numAccepted / block.num_threads() + 1;
			for(int it = 0; it < numIterations; it++){
				int index = block.num_threads() * it + block.thread_rank();

				if(index >= sh_numAccepted) continue;

				uint32_t voxelIndex = accepted[index];

				uint32_t R = voxelGrid[4 * voxelIndex + 0];
				uint32_t G = voxelGrid[4 * voxelIndex + 1];
				uint32_t B = voxelGrid[4 * voxelIndex + 2];
				uint32_t count = voxelGrid[4 * voxelIndex + 3];

				uint32_t color;
				uint8_t* rgba = (uint8_t*)&color;
				rgba[0] = R / count;
				rgba[1] = G / count;
				rgba[2] = B / count;

				int ix = voxelIndex % gridSize;
				int iy = (voxelIndex % (gridSize * gridSize)) / gridSize;
				int iz = voxelIndex / (gridSize * gridSize);

				float x = (float(ix) + 0.5f) * boxSize.x / float(gridSize);
				float y = (float(iy) + 0.5f) * boxSize.y / float(gridSize);
				float z = (float(iz) + 0.5f) * boxSize.z / float(gridSize);
				// float cubeSize = boxSize.x / float(gridSize);

				vec3 pos = {x, y, z};
				pos = pos + node->min;

				Point voxel;
				voxel.x = pos.x;
				voxel.y = pos.y;
				voxel.z = pos.z;
				voxel.color = color;

				// TODO: allocate
				node->voxels[index] = voxel;

				// since the singlecell method only projects to cells with actual geometry,
				// we can clear the voxelGrid directly here during extraction
				voxelGrid[4 * voxelIndex + 0] = 0;
				voxelGrid[4 * voxelIndex + 1] = 0;
				voxelGrid[4 * voxelIndex + 2] = 0;
				voxelGrid[4 * voxelIndex + 3] = 0;
			}

			block.sync();
		}
	}

	// update allocator.offset for subsequent allocations in "kernel.cu"
	allocator.offset = globalAllocatorOffset;

	// PRINT("smallVolumeNodeCounter: %i \n", smallVolumeNodeCounter);
	// PRINT("smallVolumePointCounter: %i k \n", (smallVolumePointCounter / 1000) );
}

};
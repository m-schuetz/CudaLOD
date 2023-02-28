

namespace sampleselect_random_blockwise_globalmem{

#include <cooperative_groups.h>
#include "lib.h.cu"
#include "methods_common.h.cu"
#include <curand_kernel.h>

// constexpr bool COMPARE_WITH_NONHEURISTIC = true;

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
	void* sssorted
){
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
	uint32_t* accepteds        = allocator.alloc<uint32_t*>(grid.num_blocks() * acceptedByteSize, "list of accepted indices");
	uint32_t* accepted         = accepteds + grid.block_rank() * acceptedCapacity;

	// Create one voxelgrid per workgroup, and a <voxelGrid> pointer that points to the active workgroup's memory
	uint64_t voxelGridByteSize = sizeof(uint32_t) * numCells;
	uint32_t* voxelGrids       = allocator.alloc<uint32_t*>(grid.num_blocks() * voxelGridByteSize, "voxel sampling grids");
	uint32_t* voxelGrid        = voxelGrids + grid.block_rank() * numCells;

	uint64_t& globalAllocatorOffset = *allocator.alloc<uint64_t*>(8);

	curandStateXORWOW_t thread_random_state;
	curand_init(grid.thread_rank(), 0, 0, &thread_random_state);

	__shared__ uint32_t sh_numAccepted;
	__shared__ uint32_t sh_workIndex;
	__shared__ uint32_t sh_iteration_block;

	// initially clear all voxel grids
	clearBuffer(voxelGrids, 0, grid.num_blocks() * voxelGridByteSize, 0);

	PRINT("gridSize: %i \n", gridSize);
	PRINT("voxelGridByteSize: %i \n", voxelGridByteSize);

	grid.sync();

	if(block.thread_rank() == 0){
		sh_iteration_block = 0;
	}

	if(isFirstThread()){
		globalAllocatorOffset = allocator.offset;

		printf("allocator.offset:        ");
		printNumber(allocator.offset, 13);
		printf("\n");
	}
	grid.sync();

	// loop until all work done, but limit loop range to be safe
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
				atomicAdd(&sh_iteration_block, 1);
			}
			block.sync();

			if(sh_workIndex >= workloadSize) break;

			Node* node = workload[sh_workIndex];
			vec3 boxSize = node->max - node->min;
			vec3 childSize = boxSize / 2.0f;

			block.sync();

			for(int childIndex = 0; childIndex < 8; childIndex++){
				Node* child = node->children[childIndex];

				if(child == nullptr) continue;
				block.sync();

				// POINTS - first-come, first-serve
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
					assert(voxelIndex >= 0);
					assert(voxelIndex < numCells);

					uint32_t randomNumber = curand(&thread_random_state);
					uint32_t encoded = 
						(randomNumber & 0b11111111'11110000'00000000'00000000) | 
						(pointIndex   & 0b00000000'00001111'11111111'11111111);

					uint32_t old = atomicMax(&voxelGrid[voxelIndex], encoded);
					bool isAccepted = (old == 0);

					if(isAccepted){
						auto targetIndex = atomicAdd(&sh_numAccepted, 1);

						accepted[targetIndex] = voxelIndex | (childIndex << 24);
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
					assert(voxelIndex >= 0);
					assert(voxelIndex < numCells);

					uint32_t randomNumber = curand(&thread_random_state);
					uint32_t encoded = 
						(randomNumber & 0b11111111'11110000'00000000'00000000) | 
						(pointIndex   & 0b00000000'00001111'11111111'11111111);

					uint32_t old = atomicMax(&voxelGrid[voxelIndex], encoded);
					bool isAccepted = (old == 0);

					if(isAccepted){
						auto targetIndex = atomicAdd(&sh_numAccepted, 1);

						accepted[targetIndex] = voxelIndex | (childIndex << 24);
					}
				}

				block.sync();
			}

			block.sync();

			// allocate memory for voxels
			Point* voxelBuffer = nullptr;
			if(block.thread_rank() == 0){
				uint64_t bufferOffset = atomicAdd(&globalAllocatorOffset, 16ull * sh_numAccepted);
				voxelBuffer = reinterpret_cast<Point*>(allocator.buffer + bufferOffset);
			}
			
			block.sync();

			if(block.thread_rank() == 0){
				node->voxels = voxelBuffer;
				node->numVoxels = sh_numAccepted;
			}

			block.sync();

			// write accepted samples to node's voxel buffer
			int numIterations = sh_numAccepted / block.num_threads() + 1;
			for(int it = 0; it < numIterations; it++){
				int index = block.num_threads() * it + block.thread_rank();

				if(index >= sh_numAccepted) continue;

				uint32_t val = accepted[index];
				uint32_t voxelIndex = (val & 0x00ffffff);
				uint32_t childIndex = (val >> 24);
				uint32_t sampleIndex = voxelGrid[voxelIndex] & 0b00000000'00001111'11111111'11111111;

				Node* child = node->children[childIndex];

				// Point voxel = child->points[sampleIndex];

				Point voxel;
				if(child->numPoints > 0){
					voxel = child->points[sampleIndex];
				}else if(child->numVoxels > 0){
					voxel = child->voxels[sampleIndex];
				}else{
					assert(false);
				}

				int ix = clamp(fGridSize * (voxel.x - node->min.x) / boxSize.x, 0.0f, fGridSize - 1.0f);
				int iy = clamp(fGridSize * (voxel.y - node->min.y) / boxSize.y, 0.0f, fGridSize - 1.0f);
				int iz = clamp(fGridSize * (voxel.z - node->min.z) / boxSize.z, 0.0f, fGridSize - 1.0f);

				voxel.x = float(ix + 0.5f) * boxSize.x / fGridSize + node->min.x;
				voxel.y = float(iy + 0.5f) * boxSize.y / fGridSize + node->min.y;
				voxel.z = float(iz + 0.5f) * boxSize.z / fGridSize + node->min.z;

				node->voxels[index] = voxel;

				// if(child->numPoints > 0){
				// 	node->voxels[index] = child->points[sampleIndex];
				// }else if(child->numVoxels > 0){
				// 	node->voxels[index] = child->voxels[sampleIndex];
				// }else{
				// 	assert(false);
				// }

				voxelGrid[voxelIndex] = 0;
			}

			block.sync();
		}
	}

	grid.sync();

	allocator.offset = globalAllocatorOffset;

	// PRINT("smallVolumeNodeCounter: %i \n", smallVolumeNodeCounter);
	// PRINT("smallVolumePointCounter: %i k \n", (smallVolumePointCounter / 1000) );


}

};
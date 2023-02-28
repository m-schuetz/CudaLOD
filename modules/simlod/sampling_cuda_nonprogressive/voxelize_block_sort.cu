

namespace voxelize_block_sort{

#include <cooperative_groups.h>
#include "lib.h.cu"
#include "methods_common.h.cu"
#include <curand_kernel.h>

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
	Point* sorted = (Point*)sssorted;
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
	uint64_t voxelGridByteSize = sizeof(uint64_t) * numCells;
	uint64_t acceptedByteSize  = sizeof(Point) * acceptedCapacity;
	uint64_t* voxelGrids       = allocator.alloc<uint64_t*>(grid.num_blocks() * voxelGridByteSize, "voxel sampling grids");
	Point* accepteds           = allocator.alloc<Point*>(grid.num_blocks() * acceptedByteSize, "list of accepted indices");
	uint64_t& globalAllocatorOffset = *allocator.alloc<uint64_t*>(8);

	uint64_t* voxelGrid        = voxelGrids + grid.block_rank() * numCells;
	Point* accepted            = accepteds + grid.block_rank() * acceptedCapacity;

	curandStateXORWOW_t thread_random_state;
	curand_init(grid.thread_rank(), 0, 0, &thread_random_state);

	__shared__ uint32_t sh_numAccepted;
	__shared__ uint32_t sh_workIndex;
	__shared__ uint32_t sh_iteration_block;
	__shared__ int3 bMin;
	__shared__ int3 bMax;

	// constexpr int hashmapCapacity = 10'000;
	// constexpr int hashmapPrime    =  9'967;
	// constexpr int hashmapPrime    =  11'003;
	constexpr int hashmapPrime    =  20'011;
	// constexpr int hashmapPrime    =  2'003;

	// constexpr int hashmapPrime    =  5'003;
	constexpr int hashmapCapacity = hashmapPrime;
	extern __shared__ int sharedMem[];
	// __shared__ uint32_t sh_hashmap[hashmapCapacity];
	uint32_t* sh_hashmap = (uint32_t*)sharedMem;

	clearBuffer(voxelGrids, 0, grid.num_blocks() * voxelGridByteSize, 0);

	grid.sync();

	if(block.thread_rank() == 0){
		sh_iteration_block = 0;
	}

	if(isFirstThread()){
		globalAllocatorOffset = allocator.offset;

		printf("allocator.offset: ");
		printNumber(allocator.offset, 10);
		printf("\n");
	}
	grid.sync();

	// return;

	// loop until all work done, but limit loop range to be safe
	for(int abc = 0; abc < 20; abc++){

		grid.sync();

		computeWorkload(nodes, numNodes, workload, workloadSize);
		if(grid.thread_rank() == 0){
			workIndexCounter = 0;
		}
		grid.sync();

		if(workloadSize == 0) break;

		// ITERATE NODES
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

			// COUNT
			block.sync();

			uint32_t countGridSize = 8;
			float fCountGridSize = countGridSize;
			uint32_t countGridCells = countGridSize * countGridSize * countGridSize;

			uint32_t* sh_countGrid = (uint32_t*)sharedMem;
			uint32_t* sh_offsetGrid = ((uint32_t*)sharedMem) + countGridCells;
			uint32_t* sh_offset = ((uint32_t*)sharedMem) + countGridCells + countGridCells;

			int numIterations_countGrid = countGridSize / block.num_threads() + 1;
			// clear
			for(int it = 0; it < numIterations_countGrid; it++){
				
				int index = block.num_threads() * it + block.thread_rank();
				if(index >= countGridCells) continue;

				sh_countGrid[index] = 0;
			}

			// ITERATE CHILD NODES
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

					int ix = fCountGridSize * (point.x - node->min.x) / boxSize.x;
					int iy = fCountGridSize * (point.y - node->min.y) / boxSize.y;
					int iz = fCountGridSize * (point.z - node->min.z) / boxSize.z;

					int voxelIndex = ix + countGridSize * iy + countGridSize * countGridSize * iz;
					voxelIndex = clamp(voxelIndex, 0, numCells - 1);
					
					atomicAdd(&sh_countGrid[voxelIndex], 1);
				}

				block.sync();

				// VOXELS
				numIterations = child->numVoxels / block.num_threads() + 1;
				for(int it = 0; it < numIterations; it++){
					int pointIndex = block.num_threads() * it + block.thread_rank();

					if(pointIndex >= child->numVoxels) continue;

					Point point = child->voxels[pointIndex];

					int ix = fCountGridSize * (point.x - node->min.x) / boxSize.x;
					int iy = fCountGridSize * (point.y - node->min.y) / boxSize.y;
					int iz = fCountGridSize * (point.z - node->min.z) / boxSize.z;

					int voxelIndex = ix + countGridSize * iy + countGridSize * countGridSize * iz;
					voxelIndex = clamp(voxelIndex, 0, numCells - 1);
					
					atomicAdd(&sh_countGrid[voxelIndex], 1);
				}

				block.sync();
			}

			block.sync();

			for(int it = 0; it < numIterations_countGrid; it++){
				
				int index = block.num_threads() * it + block.thread_rank();
				if(index >= countGridCells) continue;

				uint32_t count = sh_countGrid[index];
				uint32_t offset = atomicAdd(sh_offset, count);

				sh_offsetGrid[index] = offset;
			}

			



		}
	}

	// PRINT("smallVolumeNodeCounter: %i \n", smallVolumeNodeCounter);
	// PRINT("smallVolumePointCounter: %i k \n", (smallVolumePointCounter / 1000) );


}

};
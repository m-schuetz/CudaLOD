

namespace sampleselect_central_blockwise{

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
	uint64_t acceptedByteSize  = sizeof(Point) * acceptedCapacity;
	Point* accepteds           = allocator.alloc<Point*>(grid.num_blocks() * acceptedByteSize, "list of accepted indices");
	uint64_t& globalAllocatorOffset = *allocator.alloc<uint64_t*>(8);

	Point* accepted            = accepteds + grid.block_rank() * acceptedCapacity;

	curandStateXORWOW_t thread_random_state;
	curand_init(grid.thread_rank(), 0, 0, &thread_random_state);

	__shared__ uint32_t sh_numAccepted;
	__shared__ uint32_t sh_workIndex;
	__shared__ uint32_t sh_iteration_block;

	// prime numbers are typically good for hashmap capacities
	// maybe best to use primes that are also P = 3 mod 4?
	// (they can be used for random permutations)
	// 2'003, 9'967, 11'003, 20'011, ...
	constexpr int hashmapCapacity = 9'967;
	// Can only allocate at most 48kb static shared memory.
	// For >48kb, we'd need dynamic shared memory, but that requires
	// adjusting the launch call and praying the GPU has enough L1
	__shared__ uint32_t sh_hashmap[hashmapCapacity];

	auto hashFunction = [&](int index){
		return (index * index * 1234) % hashmapCapacity;
	};

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

			block.sync();

			int numIterationsHashmap = hashmapCapacity / block.num_threads() + 1;

			for(int childIndex = 0; childIndex < 8; childIndex++){
				Node* child = node->children[childIndex];

				if(child == nullptr) continue;
				block.sync();

				// clear hashmap
				for(int it = 0; it < numIterationsHashmap; it++){
					int index = block.num_threads() * it + block.thread_rank();

					if(index >= hashmapCapacity) continue;

					sh_hashmap[index] = 0;
				}
				
				block.sync();
				
				// POINTS
				int numIterations = child->numPoints / block.num_threads() + 1;
				for(int it = 0; it < numIterations; it++){
					int pointIndex = block.num_threads() * it + block.thread_rank();

					if(pointIndex >= child->numPoints) continue;

					Point point = child->points[pointIndex];

					float fx = fGridSize * (point.x - node->min.x) / boxSize.x;
					float fy = fGridSize * (point.y - node->min.y) / boxSize.y;
					float fz = fGridSize * (point.z - node->min.z) / boxSize.z;

					int ix = fx;
					int iy = fy;
					int iz = fz;

					float ux = fx - floor(fx) - 0.5f;
					float uy = fy - floor(fy) - 0.5f;
					float uz = fz - floor(fz) - 0.5f;
					float distance = sqrt(ux * ux + uy * uy + uz * uz);
					uint32_t idistance = 1'000.0f - 1'000.0f * distance;
					idistance = clamp(idistance, 1u, 1000u);
					idistance = idistance << 20;

					int voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;
					voxelIndex = clamp(voxelIndex, 0, numCells - 1);

					int hashmapIndex = hashFunction(voxelIndex);

					uint32_t randomNumber = curand(&thread_random_state);
					uint32_t encoded = 
						(idistance & 0b11111111'11110000'00000000'00000000) | 
						(pointIndex   & 0b00000000'00001111'11111111'11111111);

					atomicMax(&sh_hashmap[hashmapIndex], encoded);
				}

				block.sync();

				// VOXELS
				numIterations = child->numVoxels / block.num_threads() + 1;
				for(int it = 0; it < numIterations; it++){
					int pointIndex = block.num_threads() * it + block.thread_rank();

					if(pointIndex >= child->numVoxels) continue;

					Point point = child->voxels[pointIndex];

					float fx = fGridSize * (point.x - node->min.x) / boxSize.x;
					float fy = fGridSize * (point.y - node->min.y) / boxSize.y;
					float fz = fGridSize * (point.z - node->min.z) / boxSize.z;

					int ix = fx;
					int iy = fy;
					int iz = fz;

					float ux = fx - floor(fx) - 0.5f;
					float uy = fy - floor(fy) - 0.5f;
					float uz = fz - floor(fz) - 0.5f;
					float distance = ux * ux + uy * uy + uz * uz;
					uint32_t idistance = 1'000.0f - 1'000.0f * distance;
					idistance = idistance << 20;

					int voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;
					voxelIndex = clamp(voxelIndex, 0, numCells - 1);

					int hashmapIndex = hashFunction(voxelIndex);

					uint32_t encoded = 
						(idistance & 0b11111111'11110000'00000000'00000000) | 
						(pointIndex   & 0b00000000'00001111'11111111'11111111);

					atomicMax(&sh_hashmap[hashmapIndex], encoded);
				}

				block.sync();

				__shared__ int sh_count;
				sh_count = 0;
				block.sync();

				// extract into temporary accepted buffer
				for(int it = 0; it < numIterationsHashmap; it++){
					int index = block.num_threads() * it + block.thread_rank();

					if(index >= hashmapCapacity) continue;

					uint64_t encoded = sh_hashmap[index];

					if(encoded == 0) continue;

					atomicAdd(&sh_count, 1);

					uint32_t sampleIndex = encoded & 0b00000000'00001111'11111111'11111111;

					auto targetIndex = atomicAdd(&sh_numAccepted, 1);

					if(child->numPoints > 0){
						accepted[targetIndex] = child->points[sampleIndex];
					}else{
						accepted[targetIndex] = child->voxels[sampleIndex];
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

				node->voxels[index] = accepted[index];
			}

			block.sync();
		}
	}

	// PRINT("smallVolumeNodeCounter: %i \n", smallVolumeNodeCounter);
	// PRINT("smallVolumePointCounter: %i k \n", (smallVolumePointCounter / 1000) );


}

};
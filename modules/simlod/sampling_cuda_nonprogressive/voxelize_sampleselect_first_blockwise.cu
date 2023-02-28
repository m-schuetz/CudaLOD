

namespace sampleselect_first_blockwise{

#include <cooperative_groups.h>
#include "lib.h.cu"
#include "methods_common.h.cu"
#include <curand_kernel.h>

// constexpr bool COMPARE_WITH_NONHEURISTIC = true;

namespace cg = cooperative_groups;

// primes that are 
// - congruent 3 mod 4
// - the first such prime after a multiple of 128
const uint32_t Ps[391] = {
	131,263,419,523,643,787,907,1031,1163,1283,1423,1543,1667,1811,1931,2063,2179,2311,2447,2579,2699,2819,2963,3079,3203,3331,3463,3607,3719,3847,4003,4099,4231,4363,4483,4639,4751,4871,4999,5147,5279,5387,5507,5639,5779,5903,6043,6151,6287,6427,6547,6659,6791,6947,7043,7187,7307,7451,7559,7687,7823,7951,8087,8219,8363,8467,8599,8707,8839,8963,9091,9227,9371,9479,9619,9739,9859,10007,10139,10243,10391,10499,10627,10771,10883,11027,11159,11279,11399,11527,11699,11779,11923,12043,12163,12323,12451,12547,12703,12823,12959,13063,13187,13327,13451,13591,13711,13831,13963,14083,14243,14347,14479,14627,14723,14851,14983,15107,15259,15383,15511,15619,15767,15887,16007,16139,16267,16411,16519,16651,16787,16903,17027,17159,17291,17419,17539,17683,17807,17923,18059,18191,18307,18439,18583,18691,18839,18947,19079,19207,19379,19463,19603,19727,19843,19979,20107,20231,20359,20483,20611,20743,20879,21011,21139,21283,21379,21523,21647,21767,21911,22027,22147,22279,22447,22531,22679,22787,22943,23059,23203,23311,23431,23563,23687,23819,23971,24071,24203,24359,24499,24611,24763,24847,24967,25111,25219,25367,25523,25603,25747,25867,25999,26119,26251,26371,26539,26627,26759,26891,27011,27143,27271,27407,27527,27691,27779,27919,28051,28163,28307,28439,28547,28687,28807,28979,29059,29191,29327,29443,29587,29723,29851,29959,30091,30211,30347,30467,30631,30727,30851,30983,31123,31247,31379,31511,31627,31751,31883,32003,32143,32299,32411,32531,32647,32771,32911,33071,33179,33287,33427,33547,33679,33811,33923,34123,34183,34319,34439,34583,34703,34819,34963,35083,35227,35339,35491,35591,35731,35851,35983,36107,36251,36383,36523,36643,36739,36871,37003,37123,37307,37379,37507,37643,37783,37907,38039,38167,38287,38431,38543,38671,38791,38923,39043,39191,39323,39439,39563,39703,39827,39971,40087,40231,40343,40459,40583,40739,40847,41011,41131,41227,41351,41479,41603,41759,41863,41999,42131,42283,42379,42499,42643,42767,42899,43019,43151,43271,43399,43543,43651,43783,43943,44059,44171,44351,44483,44563,44683,44819,44939,45083,45191,45319,45491,45587,45707,45827,45959,46091,46219,46351,46471,46619,46723,46867,47051,47111,47251,47363,47491,47623,47779,47903,48023,48131,48259,48407,48523,48647,48779,48907,49031,49171,49307,49411,49547,49667,49807,49927,50051
};

// see https://preshing.com/20121224/how-to-generate-a-sequence-of-unique-random-integers/
// for a given number and capacity, returns a permutation to another number within
// range [0, nextPrimeAfterCapacity]
// Can be used to permutate element indices in an array,
// but you need to make sure to cap from [0, nextPrimeAfterCapacity] to [0, capacity]
uint32_t permute(int64_t number, int64_t capacity){

	if(number >= capacity) return 0xffffffff;

	// assert(number < capacity);
	int64_t pindex = (capacity - 1) / 128;

	int64_t prime = Ps[pindex];

	int64_t q = number * number;
	int64_t d = q / prime;
	int64_t residue = q - d * prime;

	int64_t result = 0;

	if(number <= prime / 2){
		result = residue;
	}else{
		result = prime - residue;
	}

	if(result < capacity){
		return result;
	}else{
		return 0xffffffff;
	}

}

// see https://preshing.com/20121224/how-to-generate-a-sequence-of-unique-random-integers/
int64_t permuteI(int64_t number, int64_t prime){

	if(number > prime){
		return number;
	}

	int64_t q = number * number;
	int64_t d = q / prime;
	int64_t residue = q - d * prime;

	if(number <= prime / 2){
		return residue;
	}else{
		return prime - residue;
	}
}

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


	// { // DEBUG

	// 	int maxval = 250;
	// 	if(grid.thread_rank() <= maxval){

	// 		uint32_t input = grid.thread_rank();
	// 		uint32_t permuted = permute(input, maxval);

	// 		printf("[%3i] => %3i \n", input, permuted);

	// 	}

	// }
	// grid.sync();


	
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

	// voxel grid mask, 1 bit per cell, indicating whether it's already occupied or still empty
	constexpr uint32_t bitgrid_size = 64;
	constexpr uint32_t bitgrid_numBits = bitgrid_size * bitgrid_size * bitgrid_size;
	constexpr uint32_t bitgrid_numU32 = bitgrid_numBits / 32;
	__shared__ uint32_t sh_bitgrid[bitgrid_numU32];

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

			// DEBUG
			// if(grid.block_rank() != 0) break;

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

			int numIterationsBitgrid = bitgrid_numU32 / block.num_threads() + 1;

			for(int childIndex = 0; childIndex < 8; childIndex++){
				Node* child = node->children[childIndex];

				if(child == nullptr) continue;
				block.sync();

				// clear bitgrid
				for(int it = 0; it < numIterationsBitgrid; it++){
					int index = block.num_threads() * it + block.thread_rank();

					if(index >= bitgrid_numU32) continue;

					sh_bitgrid[index] = 0;
				}
				
				block.sync();
				
				// POINTS - PERMUTED
				// int numIterations = child->numPoints / block.num_threads() + 1;
				// for(int it = 0; it < numIterations; it++){
				// 	int originalIndex = block.num_threads() * it + block.thread_rank();
				// 	uint32_t permutedIndex = permute(originalIndex, child->numPoints);

				// 	if(permutedIndex == 0xffffffff) continue;

				// 	assert(permutedIndex < child->numPoints);

				// 	Point point = child->points[permutedIndex];

				// 	int ix = float(bitgrid_size) * (point.x - child->min.x) / childSize.x;
				// 	int iy = float(bitgrid_size) * (point.y - child->min.y) / childSize.y;
				// 	int iz = float(bitgrid_size) * (point.z - child->min.z) / childSize.z;

				// 	uint32_t voxelIndex = ix + bitgrid_size * iy + bitgrid_size * bitgrid_size * iz;
				// 	voxelIndex = min(voxelIndex, numCells - 1);

				// 	uint32_t bitgridIndex = voxelIndex / 32;
				// 	uint32_t bitmask = 1 << (voxelIndex % 32);

				// 	uint32_t old = atomicOr(&sh_bitgrid[bitgridIndex], bitmask);
				// 	bool isAccepted = (old & bitmask) == 0;

				// 	if(isAccepted){
				// 		auto targetIndex = atomicAdd(&sh_numAccepted, 1);

				// 		accepted[targetIndex] = point;
				// 	}
				// }

				// POINTS - first-come, first-serve
				int numIterations = child->numPoints / block.num_threads() + 1;
				for(int it = 0; it < numIterations; it++){
					int pointIndex = block.num_threads() * it + block.thread_rank();

					if(pointIndex >= child->numPoints) continue;

					// uint32_t originalIndex = pointIndex;
					// uint32_t permutedIndex = permute(originalIndex, child->numPoints);

					Point point = child->points[pointIndex];

					int ix = float(bitgrid_size) * (point.x - child->min.x) / childSize.x;
					int iy = float(bitgrid_size) * (point.y - child->min.y) / childSize.y;
					int iz = float(bitgrid_size) * (point.z - child->min.z) / childSize.z;

					uint32_t voxelIndex = ix + bitgrid_size * iy + bitgrid_size * bitgrid_size * iz;
					voxelIndex = min(voxelIndex, numCells - 1);

					uint32_t bitgridIndex = voxelIndex / 32;
					uint32_t bitmask = 1 << (voxelIndex % 32);

					uint32_t old = atomicOr(&sh_bitgrid[bitgridIndex], bitmask);
					bool isAccepted = (old & bitmask) == 0;

					if(isAccepted){
						auto targetIndex = atomicAdd(&sh_numAccepted, 1);

						point.x = float(ix + 0.5f) * childSize.x / float(bitgrid_size) + child->min.x;
						point.y = float(iy + 0.5f) * childSize.y / float(bitgrid_size) + child->min.y;
						point.z = float(iz + 0.5f) * childSize.z / float(bitgrid_size) + child->min.z;
						accepted[targetIndex] = point;
					}
				}

				block.sync();

				// VOXELS
				numIterations = child->numVoxels / block.num_threads() + 1;
				for(int it = 0; it < numIterations; it++){
					int pointIndex = block.num_threads() * it + block.thread_rank();

					if(pointIndex >= child->numVoxels) continue;

					Point point = child->voxels[pointIndex];

					int ix = float(bitgrid_size) * (point.x - child->min.x) / childSize.x;
					int iy = float(bitgrid_size) * (point.y - child->min.y) / childSize.y;
					int iz = float(bitgrid_size) * (point.z - child->min.z) / childSize.z;

					uint32_t voxelIndex = ix + bitgrid_size * iy + bitgrid_size * bitgrid_size * iz;
					voxelIndex = min(voxelIndex, numCells - 1);

					uint32_t bitgridIndex = voxelIndex / 32;
					uint32_t bitmask = 1 << (voxelIndex % 32);

					uint32_t old = atomicOr(&sh_bitgrid[bitgridIndex], bitmask);
					bool isAccepted = (old & bitmask) == 0;

					if(isAccepted){
						auto targetIndex = atomicAdd(&sh_numAccepted, 1);
						point.x = float(ix + 0.5f) * childSize.x / float(bitgrid_size) + child->min.x;
						point.y = float(iy + 0.5f) * childSize.y / float(bitgrid_size) + child->min.y;
						point.z = float(iz + 0.5f) * childSize.z / float(bitgrid_size) + child->min.z;

						accepted[targetIndex] = point;
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

				// printf("numaccepted: %i \n", sh_numAccepted);
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

	grid.sync();

	allocator.offset = globalAllocatorOffset;

	// PRINT("smallVolumeNodeCounter: %i \n", smallVolumeNodeCounter);
	// PRINT("smallVolumePointCounter: %i k \n", (smallVolumePointCounter / 1000) );


}

};
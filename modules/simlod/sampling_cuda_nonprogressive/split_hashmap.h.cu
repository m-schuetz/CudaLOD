


namespace split_hashmap{

#include <cooperative_groups.h>
#include "lib.h.cu"
#include "methods_common.h.cu"

namespace cg = cooperative_groups;


void main_split(
	Allocator& allocator, 
	Box3 box, 
	Point* input_points, 
	int numPoints,
	void** nnodes,
	uint32_t* nnum_nodes,
	void** ssorted
){

	auto t_split = Timer::start("split_hashmap.h.cu");

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint32_t& numNodes = *allocator.alloc<uint32_t*>(4, "node counter");
	Node* nodes = allocator.alloc<Node*>(sizeof(Node) * 100'000, "nodes");
	Point* sorted = allocator.alloc<Point*>(sizeof(Point) * numPoints, "sorted");

	// INIT ROOT
	if(isFirstThread()){
		Node root;

		root.min = box.min;
		root.cubeSize = (box.max - box.min).longestAxis();
		root.max = box.min + root.cubeSize;

		root.cubeSize = (box.max - box.min).longestAxis();
		root.min = {0.0, 0.0, 0.0};
		root.max = root.min + root.cubeSize;

		root.level = 0;
		root.numPoints = numPoints;

		// printf("root.cubeSize: %f \n", root.cubeSize);

		nodes[0] = root;

		numNodes = 1;
	}
	Node* root = &nodes[0];
	grid.sync();



	// project to hashmap
	// constexpr uint32_t hashmapPrime    = 2'000'003;
	constexpr uint32_t hashmapPrime    = 10'000'019;
	// constexpr uint32_t hashmapPrime    = 100'003;
	constexpr uint32_t hashmapCapacity = hashmapPrime;

	uint32_t* countmap = allocator.alloc<uint32_t*>(sizeof(uint32_t) * hashmapCapacity, "counter hashmap");
	uint64_t* keymap = allocator.alloc<uint64_t*>(sizeof(uint64_t) * hashmapCapacity, "counter hashmap");

	clearBuffer(countmap, 0, sizeof(uint32_t) * hashmapCapacity, 0);
	clearBuffer(keymap, 0, sizeof(uint64_t) * hashmapCapacity, 0);

	grid.sync();

	// uint32_t pointsPerThread = numPoints / totalThreadCount + 1;
	// uint32_t pointsPerBlock  = pointsPerThread * block.num_threads();

	uint32_t& dbg_numProcessed = *allocator.alloc<uint32_t*>(4);
	dbg_numProcessed = 0;
	grid.sync();

	// constexpr int depth          = 10;
	constexpr int gridSize       = 64; // level 12: 4096, 13: 8192
	constexpr int64_t gridSize64 = gridSize; 
	constexpr float fGridSize    = gridSize;
	vec3 boxSize                 = box.size();

	uint32_t pointsPerBlock   = numPoints / block.num_threads() + 1;
	uint32_t blockFirst       = grid.block_rank() * pointsPerBlock;
	uint32_t numIterations    = pointsPerBlock / block.num_threads() + 1;

	for(uint32_t it = 0; it < numIterations; it++){

		uint32_t pointIndex = blockFirst + it * block.num_threads() + block.thread_rank();

		if(pointIndex >= numPoints) continue;

		Point point = input_points[pointIndex];

		// atomicAdd(&dbg_numProcessed, 1);

		float fx = fGridSize * ((point.x - root->min.x) / root->cubeSize);
		float fy = fGridSize * ((point.y - root->min.y) / root->cubeSize);
		float fz = fGridSize * ((point.z - root->min.z) / root->cubeSize);

		uint64_t ix = clamp(fx, 0.0f, fGridSize - 1.0f);
		uint64_t iy = clamp(fy, 0.0f, fGridSize - 1.0f);
		uint64_t iz = clamp(fz, 0.0f, fGridSize - 1.0f);

		uint64_t voxelIndex = ix + gridSize64 * iy + gridSize64 * gridSize64 * iz;
		uint64_t hashmapIndex = (voxelIndex) % hashmapCapacity;
		// uint64_t hashmapIndex = (voxelIndex * voxelIndex) % hashmapCapacity;

		// reduce than write to hashmap
		// for(int attempt = 0; attempt < 100; attempt++)
		// // int attempt = 0;
		// {
		// 	{ 
		// 		auto warp = cg::coalesced_threads();
		// 		auto group = cg::labeled_partition(warp, voxelIndex);

		// 		if(group.thread_rank() == 0){
					
		// 			uint32_t probeIndex = hashmapIndex + attempt;

		// 			uint64_t oldKey = atomicCAS(&keymap[probeIndex], 0, voxelIndex);

		// 			if(oldKey == 0 || oldKey == voxelIndex)
		// 			{
		// 				atomicAdd(&countmap[probeIndex], group.num_threads());
		// 				break;
		// 			}
		// 		}
		// 	}
		// }
		
		hashmapIndex = (voxelIndex % hashmapCapacity);

		auto warp = cg::coalesced_threads();
		// auto group = cg::labeled_partition(warp, voxelIndex);

		int pred;
		uint32_t mask = warp.match_all(voxelIndex, pred);

		if(mask == 0){
			atomicAdd(&countmap[hashmapIndex], 1);
		}else if(warp.thread_rank() == 0){
			atomicAdd(&countmap[hashmapIndex], warp.num_threads());
		}


		// int pred;
		// __match_all_sync(0xffffffff, voxelIndex, &pred);
		// // if(pred == true){
		// // 	// all same
		// // 	// if(warp.thread_rank() == 0){
		// // 	// 	atomicAdd(&countmap[hashmapIndex], warp.num_threads());
		// // 	// }
		// // }else{
		// 	atomicAdd(&countmap[hashmapIndex], 1);
		// // }

		
		// atomicAdd(&countmap[hashmapIndex], 1);
		// atomicAdd(&countmap[pointIndex % hashmapCapacity], 1);

		// write directly to hashmap
		// for(uint64_t attempt = 0; attempt < 10; attempt++){
		// 	uint64_t probeIndex = hashmapIndex + attempt;
		// 	uint64_t oldKey = atomicCAS(&keymap[probeIndex], 0, voxelIndex);

		// 	if(oldKey == 0 || oldKey == voxelIndex)
		// 	{
		// 		atomicAdd(&countmap[probeIndex], 1);
		// 		break;
		// 	}
		// }

	}

	grid.sync();

	// if(false)
	{ // DEBUG
		uint32_t& numOccupied = *allocator.alloc<uint32_t*>(4);
		uint32_t& largestCell = *allocator.alloc<uint32_t*>(4);

		if(isFirstThread()){
			numOccupied = 0;
			largestCell = 0;
		}

		grid.sync();

		processRange(0, hashmapCapacity, [&](int index){
			auto value = countmap[index];

			if(value > 0){
				atomicAdd(&numOccupied, 1);
				atomicMax(&largestCell, value);
			}
		});

		grid.sync();

		if(isFirstThread()){
			printf("numNodes: ");
			printNumber(numOccupied, 10);
			printf("\n");

			printf("largest:  ");
			printNumber(largestCell, 10);
			printf("\n");

			printf("numProcessed: ");
			printNumber(dbg_numProcessed, 10);
			printf("\n");

		}
	}




	grid.sync();

	t_split.stop();


	*nnodes = nullptr;
	*ssorted = nullptr;
	*nnum_nodes = 0;

	// *nnodes = (void*)nodes;
	// *ssorted = (void*)sorted;
	// *nnum_nodes = numNodes;

	grid.sync();

	if(false)
	if(isFirstThread())
	{
		printf("allocated memory: %i MB \n", int(allocator.offset / (1024 * 1024)));
		printf("#nodes:           %i \n", numNodes);
	}
}


};



namespace split_countsort{

#include <cooperative_groups.h>
#include "lib.h.cu"
#include "methods_common.h.cu"

namespace cg = cooperative_groups;


// splits an octree node with many points by <depth> hierachy levels
// until leaf nodes have at most MAX_POINTS_PER_NODE.
// Leaf nodes can have more points if <depth> is insufficient.
// In that case, split_node must be called again on all large leaf nodes.
void split_node(
	Allocator& allocator,
	Node* local_root,
	Point* points_unsorted,  // IN  will not be altered. Must not be same as points_sorted
	Point* points_sorted,   // OUT sorted points go here. Leaf nodes will also point here
	int depth,              // 7: 34MB, 8: 268MB, 9: 2GB! 10: 17GB!!!
	Node* nodes,
	uint32_t& numNodes
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// Save current starting position of allocator. 
	// We'll temporarily allocate some space, 
	// but we'll revert the allocation by the end of this function
	auto original_allocator_offset = allocator.offset;

	grid.sync();

	int       gridSize     = pow(2.0f, float(depth));
	float     fGridSize    = gridSize;
	int       numCells     = gridSize * gridSize * gridSize;
	uint32_t* countingGrid = allocator.alloc<uint32_t*>(sizeof(uint32_t) * numCells, "counting grid");

	clearBuffer(countingGrid, 0, sizeof(uint32_t) * numCells, 0);

	grid.sync();

	uint32_t pointsPerThread = local_root->numPoints / grid.num_threads() + 1;
	uint32_t pointsPerBlock  = pointsPerThread * block.num_threads();
	uint32_t blockFirst      = blockIdx.x * pointsPerBlock;
	Box3 box                 = {local_root->min, local_root->max};
	vec3 boxSize             = box.size();

	// PRINT("local_root->level: %i \n", local_root->level);
	// PRINT("local_root->numPoints: %i \n", local_root->numPoints);
	// PRINT("box.min: %f, %f, %f \n", box.min.x, box.min.y, box.min.z);
	// PRINT("box.max: %f, %f, %f \n", box.max.x, box.max.y, box.max.z);

	grid.sync();

	// auto t_clount_points = Timer::start("count points");

	// COUNT POINTS PER CELL
	for(int i = 0; i < pointsPerThread; i++){

		uint32_t pointIndex = blockFirst + i * block.num_threads() + threadIdx.x;

		if(pointIndex >= local_root->numPoints) break;
		
		Point point = points_unsorted[pointIndex];

		float fx = fGridSize * (point.x - box.min.x) / boxSize.x;
		float fy = fGridSize * (point.y - box.min.y) / boxSize.y;
		float fz = fGridSize * (point.z - box.min.z) / boxSize.z;

		uint32_t ix = clamp(fx, 0.0f, fGridSize - 1.0f);
		uint32_t iy = clamp(fy, 0.0f, fGridSize - 1.0f);
		uint32_t iz = clamp(fz, 0.0f, fGridSize - 1.0f);

		uint32_t voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

		// one atomicAdd per distinct voxelIndex in current warp
		// This is hardware accelerated starting with compute capability 8
		// It is not faster on compute capability 7. Capability 7 emulates it
		{ 
			auto warp = cg::coalesced_threads();
			auto group = cg::labeled_partition(warp, voxelIndex);

			if(group.thread_rank() == 0){
				atomicAdd(&countingGrid[voxelIndex], group.num_threads());
			}
		}

		// simple but slower alternative: one atomicAdd per thread in warp
		// atomicAdd(&countingGrid[voxelIndex], 1);
	}

	grid.sync();

	// auto t_20 = nanotime();

	// { // DEBUG

	// 	uint32_t& numNodes = *allocator.alloc<uint32_t*>(4);
	// 	uint32_t& largest = *allocator.alloc<uint32_t*>(4);

	// 	processRange(0, numCells, [&](int voxelIndex){
	// 		auto value = countingGrid[voxelIndex];

	// 		if(value > 0){
	// 			atomicAdd(&numNodes, 1);
	// 			atomicMax(&largest, value);
	// 		}
	// 	});

	// 	grid.sync();

	// 	PRINT("numNodes:   %i \n", numNodes);
	// 	PRINT("largest:    %i \n", largest);
	// }


	// t_clount_points.stop();

	// auto t_merge = Timer::start("merge");

	// MERGE CELLS WITH FEW POINTS
	// create grid hierarchical grids
	typedef uint32_t* ptrGrid;
	ptrGrid grids[MAX_DEPTH];

	uint32_t levelsGridSize = 1;
	for(int level = 0; level < depth; level++){
		uint32_t numCells = levelsGridSize * levelsGridSize * levelsGridSize;
		uint32_t byteSize = sizeof(uint32_t) * numCells;
		uint32_t* cg  = allocator.alloc<uint32_t*>(byteSize);

		clearBuffer(cg, 0, byteSize, 0u);

		grids[level] = cg;

		levelsGridSize = levelsGridSize * 2;
	}
	grids[depth] = countingGrid;
	grid.sync();

	// auto t_30 = nanotime();


	// do hierarchical merging
	for(int level = depth - 1; level >= 0; level--){

		int gridSize = pow(2.0f, float(level));

		processRange(0, gridSize * gridSize * gridSize, [&](int index){
			int ix = index % gridSize;
			int iy = (index % (gridSize * gridSize)) / gridSize;
			int iz = index / (gridSize * gridSize);

			int voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

			int sumPoints = 0;
			int numMergeable = 0;
			int numUnmergeable = 0;

			// process 2x2x2 cell block; manually unrolled below
			// for(int dx : {0, 1})
			// for(int dy : {0, 1})
			// for(int dz : {0, 1})
			// {
			// 	int nx = 2 * ix + dx;
			// 	int ny = 2 * iy + dy;
			// 	int nz = 2 * iz + dz;

			// 	int nVoxelIndex = nx + 2 * gridSize * ny + 4 * gridSize * gridSize * nz;

			// 	int counter = grids[level + 1][nVoxelIndex];

			// 	if(counter > 0 && counter != -1){
			// 		numMergeable++;
			// 		sumPoints += counter;
			// 	}else if(counter == -1){
			// 		numUnmergeable++;
			// 	}
			// }

			{ 
				// unrolled version of the code above.
				// makes this part of the code 2x faster (Titan RTX, CUDA 11.6, compiled with compute_86)
				int gridSize2 = 2 * gridSize;
				int c000 = grids[level + 1][((2 * ix + 0) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 0))];
				int c001 = grids[level + 1][((2 * ix + 0) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 1))];
				int c010 = grids[level + 1][((2 * ix + 0) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 0))];
				int c011 = grids[level + 1][((2 * ix + 0) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 1))];
				int c100 = grids[level + 1][((2 * ix + 1) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 0))];
				int c101 = grids[level + 1][((2 * ix + 1) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 1))];
				int c110 = grids[level + 1][((2 * ix + 1) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 0))];
				int c111 = grids[level + 1][((2 * ix + 1) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 1))];

				if(c000 == -1){ numUnmergeable++; } else if(c000 > 0) { numMergeable++; sumPoints += c000; };
				if(c001 == -1){ numUnmergeable++; } else if(c001 > 0) { numMergeable++; sumPoints += c001; };
				if(c010 == -1){ numUnmergeable++; } else if(c010 > 0) { numMergeable++; sumPoints += c010; };
				if(c011 == -1){ numUnmergeable++; } else if(c011 > 0) { numMergeable++; sumPoints += c011; };
				if(c100 == -1){ numUnmergeable++; } else if(c100 > 0) { numMergeable++; sumPoints += c100; };
				if(c101 == -1){ numUnmergeable++; } else if(c101 > 0) { numMergeable++; sumPoints += c101; };
				if(c110 == -1){ numUnmergeable++; } else if(c110 > 0) { numMergeable++; sumPoints += c110; };
				if(c111 == -1){ numUnmergeable++; } else if(c111 > 0) { numMergeable++; sumPoints += c111; };
			}
			
			if(sumPoints < MAX_POINTS_PER_NODE && sumPoints != 0 && numUnmergeable == 0){

				// MERGE
				// for(int dx : {0, 1})
				// for(int dy : {0, 1})
				// for(int dz : {0, 1})
				// {
				// 	int nx = 2 * ix + dx;
				// 	int ny = 2 * iy + dy;
				// 	int nz = 2 * iz + dz;

				// 	int nVoxelIndex = nx + 2 * gridSize * ny + 4 * gridSize * gridSize * nz;

				// 	grids[level + 1][nVoxelIndex] = 0;
				// }

				// UNROLLED MERGE
				int gridSize2 = 2 * gridSize;
				grids[level + 1][((2 * ix + 0) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 0))] = 0;
				grids[level + 1][((2 * ix + 0) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 1))] = 0;
				grids[level + 1][((2 * ix + 0) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 0))] = 0;
				grids[level + 1][((2 * ix + 0) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 1))] = 0;
				grids[level + 1][((2 * ix + 1) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 0))] = 0;
				grids[level + 1][((2 * ix + 1) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 1))] = 0;
				grids[level + 1][((2 * ix + 1) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 0))] = 0;
				grids[level + 1][((2 * ix + 1) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 1))] = 0;

				grids[level][voxelIndex] = sumPoints;
			}else if(numMergeable > 0){
				// mark as unmergeable (because it didn't fullfill merge conditions)
				grids[level][voxelIndex] = 0xffffffff;
			}else if(numUnmergeable > 0){
				// mark as unmergeable
				grids[level][voxelIndex] = 0xffffffff;
			}

		});

		grid.sync();
	}

	grid.sync();

	// auto t_40 = nanotime();

	// t_merge.stop();

	// COMPUTE LIST OF NODES
	// ALSO CREATE GRID OF POINTERS TO THESE NODES
	typedef Node* NodePtr;
	typedef Node** NodePtrGrid;
	NodePtrGrid nodePtrGrids[MAX_DEPTH];

	{
		// create and clear node pointer grids
		for(int level = 0; level <= depth; level++){

			uint32_t levelsGridSize = pow(2.0f, float(level));
			uint32_t numCells = levelsGridSize * levelsGridSize * levelsGridSize;

			NodePtr* nodeptrs = allocator.alloc<NodePtr*>(sizeof(NodePtr) * numCells, "node pointers");

			clearBuffer(nodeptrs, 0, sizeof(NodePtr) * numCells, 0u);

			nodePtrGrids[level] = nodeptrs;
		}

		grid.sync();

		// t_50 = nanotime();

		// store node pointers in grids
		{
			grid.sync();
			// auto timer = Timer::start("compute list of nodes");

			uint32_t& counter_offsets = *allocator.alloc<uint32_t*>(4, "counter offset");
			if(isFirstThread()) {
				counter_offsets = 0;
			}
			grid.sync();

			for(int level = depth; level >= 0; level--){

				grid.sync();

				int gridSize = pow(2.0f, float(level));

				processRange(0, gridSize * gridSize * gridSize, [&](int index){
					int ix = index % gridSize;
					int iy = (index % (gridSize * gridSize)) / gridSize;
					int iz = index / (gridSize * gridSize);

					int voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

					int numPoints = grids[level][voxelIndex];

					if(level == 0){
						// local root!
						for(int ox : {0, 1})
						for(int oy : {0, 1})
						for(int oz : {0, 1})
						{
							int nx = 2 * ix + ox;
							int ny = 2 * iy + oy;
							int nz = 2 * iz + oz;
							int nVoxelIndex = nx + 2 * gridSize * ny + 4 * gridSize * gridSize * nz;
							int childIndex = (ox << 2) | (oy << 1) | oz;

							Node* child = nodePtrGrids[level + 1][nVoxelIndex];

							local_root->children[childIndex] = child;
						}

						nodePtrGrids[0][0] = local_root;

					}else if(numPoints > 0){
						// leaf node
						float cubeSize = boxSize.x / float(gridSize);

						vec3 min = {
							(float(ix)) * boxSize.x / float(gridSize),
							(float(iy)) * boxSize.y / float(gridSize),
							(float(iz)) * boxSize.z / float(gridSize)
						};
						min = min + box.min;

						Node node;
						node.level       = local_root->level + level;
						node.min         = min;
						node.max         = node.min + cubeSize;
						node.voxelIndex  = voxelIndex;
						node.numPoints   = numPoints;
						node.numAdded    = 0;
						node.cubeSize    = cubeSize;
						node.pointOffset = atomicAdd(&counter_offsets, numPoints);
						node.points      = &points_sorted[node.pointOffset];

						uint32_t nodeIndex = atomicAdd(&numNodes, 1);
						nodes[nodeIndex] = node;

						nodePtrGrids[level][voxelIndex] = &nodes[nodeIndex];
					}else if(numPoints == -1){
						// inner node

						float cubeSize = boxSize.x / float(gridSize);

						vec3 min = {
							(float(ix)) * boxSize.x / float(gridSize),
							(float(iy)) * boxSize.y / float(gridSize),
							(float(iz)) * boxSize.z / float(gridSize)
						};
						min = min + box.min;

						Node node;
						node.level       = local_root->level + level;
						node.min         = min;
						node.max         = node.min + cubeSize;
						node.voxelIndex  = voxelIndex;
						node.numPoints   = 0;
						node.numAdded    = 0;
						node.cubeSize    = cubeSize;
						node.pointOffset = 0;

						for(int ox : {0, 1})
						for(int oy : {0, 1})
						for(int oz : {0, 1})
						{
							int nx = 2 * ix + ox;
							int ny = 2 * iy + oy;
							int nz = 2 * iz + oz;
							int nVoxelIndex = nx + 2 * gridSize * ny + 4 * gridSize * gridSize * nz;
							int childIndex = (ox << 2) | (oy << 1) | oz;

							Node* child = nodePtrGrids[level + 1][nVoxelIndex];

							node.children[childIndex] = child;
						}

						uint32_t nodeIndex = atomicAdd(&numNodes, 1);
						nodes[nodeIndex] = node;

						nodePtrGrids[level][voxelIndex] = &nodes[nodeIndex];
					}

				});
			}
			
			// timer.stop();
			grid.sync();
		}

		grid.sync();
	}

	grid.sync();

	// auto t_50 = nanotime();

	{ // NOW DISTRIBUTE INTO BINS/NODES
		grid.sync();
		// auto timer = Timer::start("distribute");

		uint32_t pointsPerThread = local_root->numPoints / grid.num_threads() + 1;
		uint32_t pointsPerBlock = pointsPerThread * block.num_threads();
		uint32_t blockFirst = blockIdx.x * pointsPerBlock;

		for(int i = 0; i < pointsPerThread; i++){

			uint32_t pointIndex = blockFirst + i * block.num_threads() + threadIdx.x;

			if(pointIndex >= local_root->numPoints) continue;

			Point point = points_unsorted[pointIndex];

			float fx = fGridSize * (point.x - box.min.x) / boxSize.x;
			float fy = fGridSize * (point.y - box.min.y) / boxSize.y;
			float fz = fGridSize * (point.z - box.min.z) / boxSize.z;

			uint32_t ix = clamp(fx, 0.0f, fGridSize - 1.0f);
			uint32_t iy = clamp(fy, 0.0f, fGridSize - 1.0f);
			uint32_t iz = clamp(fz, 0.0f, fGridSize - 1.0f);

			int gs = gridSize;
			for(int level = depth; level >= 0; level--){

				uint32_t voxelIndex = ix + gs * iy + gs * gs * iz;

				Node* node = nodePtrGrids[level][voxelIndex];

				if(node != nullptr){
					uint32_t localPointIndex = atomicAdd(&node->numAdded, 1);
					uint32_t pointIndex = node->pointOffset + localPointIndex;

					points_sorted[pointIndex] = point;
					break;
				}

				ix = ix / 2;
				iy = iy / 2;
				iz = iz / 2;
				gs = gs / 2;
			}

		}
		
		// timer.stop();
		grid.sync();
	}

	// auto t_60 = nanotime();


	// now revert the allocations taken during this function
	allocator.offset = original_allocator_offset;

	grid.sync();

	local_root->numPoints = 0;
	local_root->points = nullptr;

	grid.sync();

	// if(grid.thread_rank() == 0)
	// {

	// 	float millies_00_10 = double(t_10 - t_00) / 1'000'000;
	// 	float millies_10_20 = double(t_20 - t_10) / 1'000'000;
	// 	float millies_20_30 = double(t_30 - t_20) / 1'000'000;
	// 	float millies_30_40 = double(t_40 - t_30) / 1'000'000;
	// 	float millies_40_50 = double(t_50 - t_40) / 1'000'000;
	// 	float millies_50_60 = double(t_60 - t_50) / 1'000'000;

	// 	printf("=== TIMES ===\n");
	// 	printf("millies_00_10: %f \n", millies_00_10);
	// 	printf("millies_10_20: %f \n", millies_10_20);
	// 	printf("millies_20_30: %f \n", millies_20_30);
	// 	printf("millies_30_40: %f \n", millies_30_40);
	// 	printf("millies_40_50: %f \n", millies_40_50);
	// 	printf("millies_50_60: %f \n", millies_50_60);
	// }

	// t_split.stop();
	
}


void main_split(
	Allocator& allocator, 
	Box3 box, 
	Point* input_points, 
	int numPoints,
	void** nnodes,
	uint32_t* nnum_nodes,
	void** ssorted
){

	auto t_countsort = Timer::start("countsort.h.cu");

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint32_t& numNodes = *allocator.alloc<uint32_t*>(4, "node counter");
	Node* nodes        = allocator.alloc<Node*>(sizeof(Node) * 100'000, "nodes");
	Point* sorted      = allocator.alloc<Point*>(sizeof(Point) * numPoints, "sorted");

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

		nodes[0] = root;

		numNodes = 1;
	}
	grid.sync();

	// split root j
	int depth = 8;
	Node* root = &nodes[0];
	split_node(allocator, root, input_points, sorted, depth, nodes, numNodes);
	
	grid.sync();

	int oldNumNodes = numNodes;

	// if(false)
	for(int i = 0; i < oldNumNodes; i++){
		Node* node = &nodes[i];

		auto oldAllocatorOffset = allocator.offset;

		if(node->numPoints > MAX_POINTS_PER_NODE){

			// 1. COPY POINTS IN NODE INTO A TEMP BUFFER
			// 2. SPLIT WITH TEMP BUFFER AS INPUT, AND SORTED AS TARGET

			Point* tmp = allocator.alloc<Point*>(sizeof(Point) * node->numPoints);

			processRange(0, node->numPoints, [&](int index){
				tmp[index] = node->points[index];
			});

			// PRINT("SPLIT AGAIN");

			int depth = 4;
			split_node(allocator, node, tmp, node->points, depth, nodes, numNodes);

			allocator.offset = oldAllocatorOffset;

			grid.sync();
		}
	}

	t_countsort.stop();

	grid.sync();

	*nnodes = (void*)nodes;
	*ssorted = (void*)sorted;
	*nnum_nodes = numNodes;

	grid.sync();

	if(false)
	if(isFirstThread())
	{
		printf("allocated memory: %i MB \n", int(allocator.offset / (1024 * 1024)));
		printf("#nodes:           %i \n", numNodes);
	}
}


};



namespace split_countsort_blockwise{

#include <cooperative_groups.h>
#include "lib.h.cu"
#include "methods_common.h.cu"

namespace cg = cooperative_groups;

constexpr uint32_t UNMERGABLE_CELL_FLAG = 1u << 30;
constexpr uint32_t LARGE_CELL_FLAG      = 1u << 31;
constexpr uint32_t LARGE_CELLS_CAPACITY = 100'000;
constexpr uint32_t SUBGRID_DEPTH        = 4;
constexpr uint32_t SUBGRID_SIZE         = 16;
constexpr uint32_t SUBGRID_NUMCELLS     = SUBGRID_SIZE * SUBGRID_SIZE * SUBGRID_SIZE;

Points* points;
Lines* lines;
Allocator* allocator;

struct SubGrid{
	vec3 min;
	vec3 max;
	uint32_t voxelIndex;
	uint32_t numPoints;
	Node* node;
};

uint32_t* numPointsSorted;

// SUB GRIDS
uint32_t* numSubGrids;
SubGrid* subGrids;
// counters, then prefix sums
uint32_t* subGrids_l0;
uint32_t* subGrids_l1;
uint32_t* subGrids_l2;
uint32_t* subGrids_l3;
uint32_t* subGrids_l4;
// list of Node pointers
Node** subGridsPtrs_l0;
Node** subGridsPtrs_l1;
Node** subGridsPtrs_l2;
Node** subGridsPtrs_l3;
Node** subGridsPtrs_l4;

uint32_t* dbg;

struct State{
	Box3 box;
	vec3 boxSize;
	int gridSize;
	uint32_t* numNodes;
	Point* points_unsorted;
	Point* points_sorted;
};


// count how many points fall into each cell
// - we first count on the main grid (256³ if depth == 8)
// - if some cells of the main grid contain too many points, 
//   we create subgrids of depth 4 (16³ cells) and count on subgrids
// end result is a counting grid of depth 12, made up of a main grid of depth 8
// and optional sparse grids with depth 4
void doCounting(
	uint32_t* countingGrid,
	Node* local_root,
	uint32_t* largeCells,
	State state,
	Lines* lines
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint32_t pointsPerThread = local_root->numPoints / grid.num_threads() + 1;
	uint32_t pointsPerBlock  = pointsPerThread * block.num_threads();
	uint32_t blockFirst      = grid.block_rank() * pointsPerBlock;

	int gridSize             = state.gridSize;
	float fGridSize          = state.gridSize;
	Box3 box                 = state.box;
	vec3 boxSize             = state.boxSize;
	Point* points_unsorted   = state.points_unsorted;
	
	// count points on main counting grid (if 8 levels -> 256³)
	for(int i = 0; i < pointsPerThread; i++){

		uint32_t pointIndex = blockFirst + i * block.num_threads() + block.thread_rank();

		if(pointIndex >= local_root->numPoints) break;
		
		Point point = state.points_unsorted[pointIndex];

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
		// { 
		// 	auto warp = cg::coalesced_threads();
		// 	auto group = cg::labeled_partition(warp, voxelIndex);

		// 	if(group.thread_rank() == 0){
		// 		atomicAdd(&countingGrid[voxelIndex], group.num_threads());
		// 	}
		// }

		// simple but slower alternative: one atomicAdd per thread in warp
		uint32_t old = atomicAdd(&countingGrid[voxelIndex], 1);

		// cells with too many points are noted and added to a list
		if(old == MAX_POINTS_PER_NODE){
			uint32_t largeCellIndex = atomicAdd(numSubGrids, 1);
			largeCells[largeCellIndex] = voxelIndex;
		}
	}

	grid.sync();

	// create sub count grids with up to 16³ size for all cells with large counters.
	subGrids = allocator->alloc<SubGrid*>(*numSubGrids * sizeof(SubGrid));
	subGrids_l0 = allocator->alloc<uint32_t*>(*numSubGrids *  1 *  1 *  1 * sizeof(uint32_t));
	subGrids_l1 = allocator->alloc<uint32_t*>(*numSubGrids *  2 *  2 *  2 * sizeof(uint32_t));
	subGrids_l2 = allocator->alloc<uint32_t*>(*numSubGrids *  4 *  4 *  4 * sizeof(uint32_t));
	subGrids_l3 = allocator->alloc<uint32_t*>(*numSubGrids *  8 *  8 *  8 * sizeof(uint32_t));
	subGrids_l4 = allocator->alloc<uint32_t*>(*numSubGrids * 16 * 16 * 16 * sizeof(uint32_t));

	clearBuffer(subGrids, 0, *numSubGrids * sizeof(SubGrid), 0);
	clearBuffer(subGrids_l0, 0, *numSubGrids *  1 *  1 *  1 * sizeof(uint32_t), 0);
	clearBuffer(subGrids_l1, 0, *numSubGrids *  2 *  2 *  2 * sizeof(uint32_t), 0);
	clearBuffer(subGrids_l2, 0, *numSubGrids *  4 *  4 *  4 * sizeof(uint32_t), 0);
	clearBuffer(subGrids_l3, 0, *numSubGrids *  8 *  8 *  8 * sizeof(uint32_t), 0);
	clearBuffer(subGrids_l4, 0, *numSubGrids * 16 * 16 * 16 * sizeof(uint32_t), 0);

	grid.sync();

	// for large cells, replace counters in main grid with indices to respective sub grids
	processRange(0, *numSubGrids, [&](int index){
		uint32_t voxelIndex = largeCells[index];

		int ix = voxelIndex % gridSize;
		int iy = (voxelIndex % (gridSize * gridSize)) / gridSize;
		int iz = voxelIndex / (gridSize * gridSize);

		auto& subgrid = subGrids[index];
		subgrid.min = {
			boxSize.x * (float(ix)) / fGridSize + box.min.x,
			boxSize.y * (float(iy)) / fGridSize + box.min.y,
			boxSize.z * (float(iz)) / fGridSize + box.min.z
		};
		subgrid.max = {
			boxSize.x * (float(ix) + 1.0f) / fGridSize + box.min.x,
			boxSize.y * (float(iy) + 1.0f) / fGridSize + box.min.y,
			boxSize.z * (float(iz) + 1.0f) / fGridSize + box.min.z
		};
		subgrid.voxelIndex = voxelIndex;
		subgrid.numPoints = countingGrid[voxelIndex];

		uint32_t flagAndPointer = LARGE_CELL_FLAG | index;

		countingGrid[voxelIndex] = flagAndPointer;
	});

	grid.sync();

	// count again, but this time on the generated subgrids
	// - project all cells again to main grid
	// - points that fall in large cells then project to the respective subgrid
	if((*numSubGrids) > 0)
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

		uint32_t counter = countingGrid[voxelIndex];

		bool isLargeCell = (counter & LARGE_CELL_FLAG) != 0;
		if(isLargeCell){
			// count this point on the sub-count-grid
			float s_fx = 16.0f * fmodf(fx, 1.0f);
			float s_fy = 16.0f * fmodf(fy, 1.0f);
			float s_fz = 16.0f * fmodf(fz, 1.0f);

			uint32_t s_ix = min(max(int(s_fx), 0), 15);
			uint32_t s_iy = min(max(int(s_fy), 0), 15);
			uint32_t s_iz = min(max(int(s_fz), 0), 15);

			uint32_t s_voxelIndex = s_ix + 16 * s_iy + 16 * 16 * s_iz;

			uint32_t countGridIndex = counter & 0x0fffffff;
			uint32_t* subGridData = subGrids_l4 + countGridIndex * SUBGRID_NUMCELLS;
			atomicAdd(&subGridData[s_voxelIndex], 1);
		}
	}

	grid.sync();
}

// each cell represents an octree node, but we want to avoid nodes with too litle points
// this function merges cells with few points, provided the merged cell has less than 
// the threshold
void mergeSubGrids(){
	auto grid = cg::this_grid();

	// NOTE: each subgrid buffer contains <numSubGrids> grids made of gridSize³ cells, each.
	// to access the subgrid for subgrid[i] at <level>, compute:
	//    grids[level] + i * gridSize³, where gridSize = pow(2, level)³
	uint32_t* grids[5];
	grids[0] = subGrids_l0;
	grids[1] = subGrids_l1;
	grids[2] = subGrids_l2;
	grids[3] = subGrids_l3;
	grids[4] = subGrids_l4;

	// we start iterating from the second deepest level, 
	// and check each cell if we can merge the corresponding 
	// 8 cells from the next higher level
	for(int level : {3, 2, 1, 0}){

		grid.sync();

		int gridSize  = pow(2.0f, float(level));
		float fGridSize = gridSize;
		int gridSize2 = 2 * gridSize;

		processRange(0, (*numSubGrids) * gridSize * gridSize * gridSize, [&](int index){

			int subgridIndex = index / (gridSize * gridSize * gridSize);
			int voxelIndex = index % (gridSize * gridSize * gridSize);

			auto& subGrid = subGrids[subgridIndex];

			float voxelSize = (subGrid.max - subGrid.min).x / fGridSize;

			int ix = voxelIndex % gridSize;
			int iy = voxelIndex % (gridSize * gridSize) / gridSize;
			int iz = voxelIndex / (gridSize * gridSize);

			// counters from this and next level
			uint32_t* counters_this = grids[level + 0] + subgridIndex * gridSize * gridSize * gridSize;
			uint32_t* counters_next = grids[level + 1] + subgridIndex * gridSize2 * gridSize2 * gridSize2;
			
			// retrieve the 8 counters from the next higher level
			uint32_t c000 = counters_next[((2 * ix + 0) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 0))];
			uint32_t c001 = counters_next[((2 * ix + 0) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 1))];
			uint32_t c010 = counters_next[((2 * ix + 0) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 0))];
			uint32_t c011 = counters_next[((2 * ix + 0) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 1))];
			uint32_t c100 = counters_next[((2 * ix + 1) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 0))];
			uint32_t c101 = counters_next[((2 * ix + 1) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 1))];
			uint32_t c110 = counters_next[((2 * ix + 1) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 0))];
			uint32_t c111 = counters_next[((2 * ix + 1) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 1))];

			int sumPoints = 0;
			int numMergeable = 0;
			int numUnmergeable = 0;

			// and check whether we can merge them
			if(c000 == 0xffffffff){ numUnmergeable++; } else if(c000 > 0) { numMergeable++; sumPoints += c000; };
			if(c001 == 0xffffffff){ numUnmergeable++; } else if(c001 > 0) { numMergeable++; sumPoints += c001; };
			if(c010 == 0xffffffff){ numUnmergeable++; } else if(c010 > 0) { numMergeable++; sumPoints += c010; };
			if(c011 == 0xffffffff){ numUnmergeable++; } else if(c011 > 0) { numMergeable++; sumPoints += c011; };
			if(c100 == 0xffffffff){ numUnmergeable++; } else if(c100 > 0) { numMergeable++; sumPoints += c100; };
			if(c101 == 0xffffffff){ numUnmergeable++; } else if(c101 > 0) { numMergeable++; sumPoints += c101; };
			if(c110 == 0xffffffff){ numUnmergeable++; } else if(c110 > 0) { numMergeable++; sumPoints += c110; };
			if(c111 == 0xffffffff){ numUnmergeable++; } else if(c111 > 0) { numMergeable++; sumPoints += c111; };

			if(sumPoints < MAX_POINTS_PER_NODE && sumPoints != 0 && numUnmergeable == 0){
				// MERGE

				counters_next[((2 * ix + 0) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 0))] = 0;
				counters_next[((2 * ix + 0) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 1))] = 0;
				counters_next[((2 * ix + 0) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 0))] = 0;
				counters_next[((2 * ix + 0) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 1))] = 0;
				counters_next[((2 * ix + 1) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 0))] = 0;
				counters_next[((2 * ix + 1) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 1))] = 0;
				counters_next[((2 * ix + 1) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 0))] = 0;
				counters_next[((2 * ix + 1) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 1))] = 0;

				counters_this[voxelIndex] = sumPoints;
			}else if(numMergeable > 0){
				// mark as unmergeable because it didn't fullfill merge conditions
				counters_this[voxelIndex] = 0xffffffff;
			}else if(numUnmergeable > 0){
				// mark as unmergeable
				counters_this[voxelIndex] = 0xffffffff;
			}
		});
	}
}

// same procedure as merging the subgrids
void mergeMainGrid(
	int depth, 
	uint32_t** grids
){

	auto grid = cg::this_grid();

	// do hierarchical merging
	for(int level = depth - 1; level >= 0; level--){

		int gridSize = pow(2.0f, float(level));
		int gridSize2 = 2 * gridSize;

		processRange(0, gridSize * gridSize * gridSize, [&](int index){
			int ix = index % gridSize;
			int iy = (index % (gridSize * gridSize)) / gridSize;
			int iz = index / (gridSize * gridSize);

			int voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;
			
			// retrieve the 8 counters from the next higher level
			uint32_t c000 = grids[level + 1][((2 * ix + 0) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 0))];
			uint32_t c001 = grids[level + 1][((2 * ix + 0) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 1))];
			uint32_t c010 = grids[level + 1][((2 * ix + 0) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 0))];
			uint32_t c011 = grids[level + 1][((2 * ix + 0) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 1))];
			uint32_t c100 = grids[level + 1][((2 * ix + 1) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 0))];
			uint32_t c101 = grids[level + 1][((2 * ix + 1) + gridSize2 * (2 * iy + 0) + gridSize2 * gridSize2 * (2 * iz + 1))];
			uint32_t c110 = grids[level + 1][((2 * ix + 1) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 0))];
			uint32_t c111 = grids[level + 1][((2 * ix + 1) + gridSize2 * (2 * iy + 1) + gridSize2 * gridSize2 * (2 * iz + 1))];

			int sumPoints = 0;
			int numMergeable = 0;
			int numUnmergeable = 0;

			auto isUnmergable = [](uint32_t value){
				if(value == 0xffffffff) return true;
				if((value & LARGE_CELL_FLAG) == LARGE_CELL_FLAG) return true;
				return false;
			};

			// and check whether we can merge them
			if(isUnmergable(c000)){ numUnmergeable++; } else if(c000 > 0) { numMergeable++; sumPoints += c000; };
			if(isUnmergable(c001)){ numUnmergeable++; } else if(c001 > 0) { numMergeable++; sumPoints += c001; };
			if(isUnmergable(c010)){ numUnmergeable++; } else if(c010 > 0) { numMergeable++; sumPoints += c010; };
			if(isUnmergable(c011)){ numUnmergeable++; } else if(c011 > 0) { numMergeable++; sumPoints += c011; };
			if(isUnmergable(c100)){ numUnmergeable++; } else if(c100 > 0) { numMergeable++; sumPoints += c100; };
			if(isUnmergable(c101)){ numUnmergeable++; } else if(c101 > 0) { numMergeable++; sumPoints += c101; };
			if(isUnmergable(c110)){ numUnmergeable++; } else if(c110 > 0) { numMergeable++; sumPoints += c110; };
			if(isUnmergable(c111)){ numUnmergeable++; } else if(c111 > 0) { numMergeable++; sumPoints += c111; };

			if(sumPoints < MAX_POINTS_PER_NODE && sumPoints != 0 && numUnmergeable == 0){
				// MERGE

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
}

// transform the counters into pointers to octree nodes
// matches the prefix sum step of counting sort, 
// but the prefix sums are stored in node->pointOffset.
// Additionally, node->points is made to directly point to
// the memory location that pointOffset references
void createSubPointers(
	int depth, 
	Node* nodes,
	Point* points_sorted,
	uint32_t& numNodes
){

	auto grid = cg::this_grid();

	uint32_t* countgrids[5];
	countgrids[0] = subGrids_l0;
	countgrids[1] = subGrids_l1;
	countgrids[2] = subGrids_l2;
	countgrids[3] = subGrids_l3;
	countgrids[4] = subGrids_l4;

	subGridsPtrs_l0 = allocator->alloc<Node**>((*numSubGrids) *  1 *  1 *  1 * sizeof(Node*));
	subGridsPtrs_l1 = allocator->alloc<Node**>((*numSubGrids) *  2 *  2 *  2 * sizeof(Node*));
	subGridsPtrs_l2 = allocator->alloc<Node**>((*numSubGrids) *  4 *  4 *  4 * sizeof(Node*));
	subGridsPtrs_l3 = allocator->alloc<Node**>((*numSubGrids) *  8 *  8 *  8 * sizeof(Node*));
	subGridsPtrs_l4 = allocator->alloc<Node**>((*numSubGrids) * 16 * 16 * 16 * sizeof(Node*));

	Node** nodegrids[5];
	nodegrids[0] = subGridsPtrs_l0;
	nodegrids[1] = subGridsPtrs_l1;
	nodegrids[2] = subGridsPtrs_l2;
	nodegrids[3] = subGridsPtrs_l3;
	nodegrids[4] = subGridsPtrs_l4;

	uint32_t& counter_offsets = *numPointsSorted;
	grid.sync();

	for(int sublevel : {4, 3, 2, 1, 0}){

		grid.sync();

		int gridSize  = pow(2.0f, float(sublevel));

		processRange(0, (*numSubGrids) * gridSize * gridSize * gridSize, [&](int index){

			int subgridIndex = index / (gridSize * gridSize * gridSize);
			int voxelIndex = index % (gridSize * gridSize * gridSize);
			
			auto& subGrid = subGrids[subgridIndex];

			Box3 box = {subGrid.min, subGrid.max};
			vec3 boxSize = box.size();
			float cubeSize = boxSize.x / float(gridSize);

			int ix = voxelIndex % gridSize;
			int iy = voxelIndex % (gridSize * gridSize) / gridSize;
			int iz = voxelIndex / (gridSize * gridSize);

			vec3 min = {
				(float(ix)) * boxSize.x / float(gridSize),
				(float(iy)) * boxSize.y / float(gridSize),
				(float(iz)) * boxSize.z / float(gridSize)
			};
			min = min + box.min;
			vec3 max = min + cubeSize;

			uint32_t* countgrid = countgrids[sublevel] + subgridIndex * gridSize * gridSize * gridSize;
			Node** nodegrid = nodegrids[sublevel] + subgridIndex * gridSize * gridSize * gridSize;

			int numPoints = countgrid[voxelIndex];

			if(numPoints > 0){
				// leaf node

				Node node;
				node.level       = depth + sublevel;
				node.min         = min;
				node.max         = max;
				node.voxelIndex  = voxelIndex;
				node.numPoints   = numPoints;
				node.numAdded    = 0;
				node.cubeSize    = cubeSize;
				node.pointOffset = atomicAdd(&counter_offsets, numPoints);
				node.points      = &points_sorted[node.pointOffset];
				node.dbg         = subGrid.voxelIndex;

				uint32_t nodeIndex = atomicAdd(&numNodes, 1);
				nodes[nodeIndex] = node;
				nodegrid[voxelIndex] = &nodes[nodeIndex];

			}else if(numPoints == 0xffffffff){
				// inner node

				Node** nodegrid_next = nodegrids[sublevel + 1] + subgridIndex * 8 * gridSize * gridSize * gridSize;

				Node node;
				node.level       = depth + sublevel;
				node.min         = min;
				node.max         = max;
				node.voxelIndex  = voxelIndex;
				node.numPoints   = 0;
				node.numAdded    = 0;
				node.cubeSize    = cubeSize;
				node.pointOffset = 0;
				node.dbg         = subGrid.voxelIndex;

				for(int ox : {0, 1})
				for(int oy : {0, 1})
				for(int oz : {0, 1})
				{
					int nx = 2 * ix + ox;
					int ny = 2 * iy + oy;
					int nz = 2 * iz + oz;
					int nVoxelIndex = nx + 2 * gridSize * ny + 4 * gridSize * gridSize * nz;
					int childIndex = (ox << 2) | (oy << 1) | oz;

					Node* child = nodegrid_next[nVoxelIndex];

					node.children[childIndex] = child;
				}

				uint32_t nodeIndex = atomicAdd(&numNodes, 1);
				nodes[nodeIndex] = node;
				nodegrid[voxelIndex] = &nodes[nodeIndex];

				// subgrid root node
				if(sublevel == 0){
					subGrid.node = &nodes[nodeIndex];
				}
			}else{
				nodegrid[voxelIndex] = NULL;
			}

		});

	}

	grid.sync();
}


void createMainPointers(
	int depth, 
	Node*** nodegrids, 
	uint32_t** grids,
	Node* local_root,
	Node* nodes,
	Box3 box,
	vec3 boxSize,
	Point* points_sorted,
	uint32_t& numNodes
){

	auto grid = cg::this_grid();

	// store node pointers in grids
	{
		uint32_t& counter_offsets = *numPointsSorted;
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

				bool hasSubgrid = (level == depth) && ((numPoints & LARGE_CELL_FLAG) != 0);

				if(hasSubgrid){
					// if this cell has a subgrid, then point to that subgrids root node
					uint64_t subgridIndex = numPoints & 0x0fffffff;
					nodegrids[level][voxelIndex] = subGrids[subgridIndex].node;
				}
				else if(level == 0)
				{
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

						Node* child = nodegrids[level + 1][nVoxelIndex];

						local_root->children[childIndex] = child;
					}

					nodegrids[0][0] = local_root;

				}else if(numPoints == 0xffffffff){
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

						Node* child = nodegrids[level + 1][nVoxelIndex];

						node.children[childIndex] = child;
					}

					uint32_t nodeIndex = atomicAdd(&numNodes, 1);
					nodes[nodeIndex] = node;

					nodegrids[level][voxelIndex] = &nodes[nodeIndex];
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

					nodegrids[level][voxelIndex] = &nodes[nodeIndex];
				}

			});
		}
		
		grid.sync();
	}

	grid.sync();
}

// find the octree node that the given point belongs to, and add the point to that node
void distributeSubGrid(
	Point point, 
	uint32_t 
	subgridIndex, 
	SubGrid& subgrid, 
	Point* points_sorted,
	vec3 gridPos
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	vec3 boxSize = subgrid.max - subgrid.min;

	Node** nodegrids[5];
	nodegrids[0] = subGridsPtrs_l0;
	nodegrids[1] = subGridsPtrs_l1;
	nodegrids[2] = subGridsPtrs_l2;
	nodegrids[3] = subGridsPtrs_l3;
	nodegrids[4] = subGridsPtrs_l4;

	// Although the math here is correct, 
	// the results can be wrong due to floating-point behaviour
	// We need to compute subgrid voxel indices exactly 
	// the same way as in the doCounting function
	// float fx = SUBGRID_SIZE * (point.x - subgrid.min.x) / boxSize.x;
	// float fy = SUBGRID_SIZE * (point.y - subgrid.min.y) / boxSize.y;
	// float fz = SUBGRID_SIZE * (point.z - subgrid.min.z) / boxSize.z;
	// uint32_t ix = clamp(fx, 0.0f, SUBGRID_SIZE - 1.0f);
	// uint32_t iy = clamp(fy, 0.0f, SUBGRID_SIZE - 1.0f);
	// uint32_t iz = clamp(fz, 0.0f, SUBGRID_SIZE - 1.0f);

	// Same math as above, but correct because it's identical to 
	// how we computed things in doCounting, thus we get the same 
	// roundings in the floats, thus the same voxelIndex results.
	float s_fx = 16.0f * fmodf(gridPos.x, 1.0f);
	float s_fy = 16.0f * fmodf(gridPos.y, 1.0f);
	float s_fz = 16.0f * fmodf(gridPos.z, 1.0f);

	uint32_t ix = min(max(int(s_fx), 0), 15);
	uint32_t iy = min(max(int(s_fy), 0), 15);
	uint32_t iz = min(max(int(s_fz), 0), 15);

	int gs = SUBGRID_SIZE;
	for(int level : {4, 3, 2, 1, 0}){
		uint32_t voxelIndex = ix + gs * iy + gs * gs * iz;

		Node** nodegrid = nodegrids[level] + subgridIndex * gs * gs * gs;
		Node* node = nodegrid[voxelIndex];

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

// distribute points to octree nodes.
void distribute(
	Node* local_root, 
	Point* points_unsorted, 
	Point* points_sorted, 
	Box3 box, 
	vec3 boxSize, 
	int gridSize,
	int depth,
	uint32_t** countgrids,
	Node*** nodegrids
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	float fGridSize = gridSize;

	uint32_t pointsPerThread = local_root->numPoints / grid.num_threads() + 1;
	uint32_t pointsPerBlock = pointsPerThread * block.num_threads();
	uint32_t blockFirst = blockIdx.x * pointsPerBlock;

	for(int i = 0; i < pointsPerThread; i++){

		uint32_t pointIndex = blockFirst + i * block.num_threads() + block.thread_rank();

		if(pointIndex >= local_root->numPoints) continue;

		Point point = points_unsorted[pointIndex];

		float fx = fGridSize * (point.x - box.min.x) / boxSize.x;
		float fy = fGridSize * (point.y - box.min.y) / boxSize.y;
		float fz = fGridSize * (point.z - box.min.z) / boxSize.z;

		uint32_t ix = clamp(fx, 0.0f, fGridSize - 1.0f);
		uint32_t iy = clamp(fy, 0.0f, fGridSize - 1.0f);
		uint32_t iz = clamp(fz, 0.0f, fGridSize - 1.0f);

		int gs = gridSize;
		for(int level = depth; level >= 0; level--)
		// for(int level : {depth, depth - 1, depth - 2})
		{

			uint32_t voxelIndex = ix + gs * iy + gs * gs * iz;

			Node* node = nodegrids[level][voxelIndex];

			// check if the target cell has a subgrid (because too many points)
			uint32_t counter = countgrids[level][voxelIndex];
			bool isLargeCell = (level == depth) && ((counter & LARGE_CELL_FLAG) == LARGE_CELL_FLAG);
			
			if(isLargeCell){
				// target cell has a subgrid, let's use the subgrid to find the correct octree node
				uint32_t subgridIndex = counter & 0x0fffffff;
				auto& subgrid = subGrids[subgridIndex];

				distributeSubGrid(point, subgridIndex, subgrid, points_sorted, {fx, fy, fz});

				break;
			}else if(node != nullptr){
				uint32_t localPointIndex = atomicAdd(&node->numAdded, 1);
				uint32_t pointIndex = node->pointOffset + localPointIndex;

				if(pointIndex >= local_root->numPoints){

					if(pointIndex == local_root->numPoints)
						printf("error, point index too big: %i \n", pointIndex);

					break;
				}

				points_sorted[pointIndex] = point;
				break;
			}

			ix = ix / 2;
			iy = iy / 2;
			iz = iz / 2;
			gs = gs / 2;
		}

	}
	
	grid.sync();
}

// check that the sum of counters is equal to the number of points
void dbg_verifyCounters(
	int depth,
	uint32_t** countgrids,
	Node*** nodegrids
){
	auto grid = cg::this_grid();

	// {
	// 	grid.sync();
	// 	processRange(0, 10, [&](int index){
	// 		auto& subgrid = subGrids[index];
	// 		printf("subgrid: voxelIndex: %i; #points: %i \n", subgrid.voxelIndex, subgrid.numPoints);
	// 	});
	// }

	grid.sync();
	uint32_t& pointcount_main = *allocator->alloc<uint32_t*>(4);
	uint32_t& nodecount_main  = *allocator->alloc<uint32_t*>(4);
	uint32_t& pointcount_sub  = *allocator->alloc<uint32_t*>(4);
	uint32_t& nodecount_sub   = *allocator->alloc<uint32_t*>(4);
	uint32_t& nodecount_main_inner = *allocator->alloc<uint32_t*>(4);
	uint32_t& nodecount_main_leaf  = *allocator->alloc<uint32_t*>(4);
	uint32_t& nodecount_sub_inner  = *allocator->alloc<uint32_t*>(4);
	uint32_t& nodecount_sub_leaf   = *allocator->alloc<uint32_t*>(4);

	pointcount_main = 0;
	nodecount_main  = 0;
	pointcount_sub  = 0;
	nodecount_sub   = 0;

	nodecount_main_inner = 0;
	nodecount_main_leaf = 0;
	nodecount_sub_inner = 0;
	nodecount_sub_leaf = 0;

	PRINT("=== dbg_verifyCounters ===\n");

	{ // main grid
		grid.sync();

		for(int level = 0; level <= depth; level++){

			int gridSize = pow(2.0f, float(level));
			uint32_t* countgrid = countgrids[level];

			processRange(0, gridSize * gridSize * gridSize, [&](int index){
				uint32_t count = countgrid[index];

				bool isLargeCell = (level == depth) && ((count & LARGE_CELL_FLAG) == LARGE_CELL_FLAG);

				if(!isLargeCell){
					if(count == 0xffffffff){
						atomicAdd(&nodecount_main, 1);
						atomicAdd(&nodecount_main_inner, 1);
					}else if(count != 0){
						atomicAdd(&pointcount_main, count);
						atomicAdd(&nodecount_main, 1);
						atomicAdd(&nodecount_main_leaf, 1);
					}
				}
				
			});
		}
		grid.sync(); 
	}

	{ // sub grid

		uint32_t* grids[5];
		grids[0] = subGrids_l0;
		grids[1] = subGrids_l1;
		grids[2] = subGrids_l2;
		grids[3] = subGrids_l3;
		grids[4] = subGrids_l4;

		for(int level : {4, 3, 2, 1, 0}){

			uint32_t* subgrid = grids[level];

			grid.sync();

			int gridSize  = pow(2.0f, float(level));

			processRange(0, (*numSubGrids) * gridSize * gridSize * gridSize, [&](int index){
				uint32_t count = subgrid[index];

				if(count == 0xffffffff){
					atomicAdd(&nodecount_sub, 1);
					atomicAdd(&nodecount_sub_inner, 1);
				}else if(count != 0){
					atomicAdd(&pointcount_sub, count);
					atomicAdd(&nodecount_sub, 1);
					atomicAdd(&nodecount_sub_leaf, 1);
				}
			});
			
		}
	}

	grid.sync();

	// PRINT("expected:         %i \n", pointcount_main);
	PRINT("==========================\n");
	PRINT("#large cells:     %8i \n", *numSubGrids);
	PRINT("pointcount[main]: %8i \n", pointcount_main);
	PRINT("pointcount[sub]:  %8i \n", pointcount_sub);
	PRINT("total:            %8i \n", pointcount_main + pointcount_sub);
	PRINT("nodecount[main]:  %8i \n", nodecount_main);
	PRINT("    inner:        %8i \n", nodecount_main_inner);
	PRINT("    leaves:       %8i \n", nodecount_main_leaf);
	PRINT("nodecount[sub]:   %8i \n", nodecount_sub);
	PRINT("    inner:        %8i \n", nodecount_sub_inner);
	PRINT("    leaves:       %8i \n", nodecount_sub_leaf);
	PRINT("total:            %8i \n", nodecount_main + nodecount_sub);
	PRINT("==========================\n");
}


// splits an octree node with many points by <depth> hierachy levels
// until leaf nodes have at most MAX_POINTS_PER_NODE.
// Leaf nodes can have more points if <depth> is insufficient.
// In that case, split_node must be called again on all large leaf nodes.
void split_node(
	Node* local_root,
	Point* points_unsorted,  // IN  will not be altered. Must not be same as points_sorted
	Point* points_sorted,   // OUT sorted points go here. Leaf nodes will also point here
	int depth,              // 7: 34MB, 8: 268MB, 9: 2GB! 10: 17GB!!!
	Node* nodes,
	uint32_t& numNodes,
	Lines* lines
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// Save current starting position of allocator. 
	// We'll temporarily allocate some space, 
	// but we'll revert the allocation by the end of this function
	auto original_allocator_offset = allocator->offset;

	grid.sync();

	int gridSize = pow(2.0f, float(depth));

	uint32_t pointsPerThread = local_root->numPoints / grid.num_threads() + 1;
	uint32_t pointsPerBlock  = pointsPerThread * block.num_threads();
	Box3 box                 = {local_root->min, local_root->max};
	vec3 boxSize             = box.size();

	// INIT MAIN GRIDS
	uint32_t** countgrids = allocator->alloc<uint32_t**>((depth + 1) * sizeof(uint32_t*));
	// a list of grids of node pointers...
	Node*** nodegrids = allocator->alloc<Node***>((depth + 1) * sizeof(Node**));

	for(int i = 0; i <= depth; i++){
		int gridSize = pow(2.0f, float(i));
		
		// countgrid
		uint32_t* countgrid_i = allocator->alloc<uint32_t*>(gridSize * gridSize * gridSize * sizeof(uint32_t));
		countgrids[i] = countgrid_i;
		clearBuffer(countgrid_i, 0, gridSize * gridSize * gridSize * sizeof(uint32_t), 0);

		// nodegrid
		Node** nodegrid_i = allocator->alloc<Node**>(gridSize * gridSize * gridSize * sizeof(Node*));
		nodegrids[i] = nodegrid_i;
		clearBuffer(nodegrid_i, 0, gridSize * gridSize * gridSize * sizeof(Node*), 0);
	}
	uint32_t* countingGrid = countgrids[depth];
	grid.sync();
	
	uint32_t* largeCells = allocator->alloc<uint32_t*>(4 * LARGE_CELLS_CAPACITY);
	numSubGrids = allocator->alloc<uint32_t*>(4);
	numPointsSorted = allocator->alloc<uint32_t*>(4);
	*numSubGrids = 0;
	*numPointsSorted = 0;

	State state;
	state.box = box;
	state.boxSize = boxSize;
	state.gridSize = gridSize;
	state.numNodes = &numNodes;
	state.points_unsorted = points_unsorted;
	state.points_sorted = points_sorted;
	
	grid.sync();

	doCounting(countingGrid, local_root, largeCells, state, lines); grid.sync();

	// dbg_verifyCounters(depth, countgrids, nodegrids); grid.sync();

	mergeSubGrids(); grid.sync();
	mergeMainGrid(depth, countgrids); grid.sync();
	// dbg_verifyCounters(depth, countgrids, nodegrids); grid.sync();
	createSubPointers(depth, nodes, points_sorted, numNodes); grid.sync();
	createMainPointers(depth, nodegrids, countgrids, local_root, nodes, box, boxSize, points_sorted, numNodes); grid.sync();

	// // DEBUG DRAW SUBGRID ROOTS
	// grid.sync();
	// processRange(0, *numSubGrids, [&](int index){
	// 	auto& subgrid = subGrids[index];
	// 	Node* node = subgrid.node;

	// 	vec3 pos = (node->max + node->min) / 2.0f;
	// 	vec3 size = node->max - node->min;

	// 	// if(subgrid.voxelIndex > 10 && subgrid.voxelIndex < 100)
	// 	// drawBoundingBox(lines, pos, size, 0x00ff00ff);

	// 	if(subgrid.voxelIndex == 29 || subgrid.voxelIndex == 45){
	// 		drawBoundingBox(lines, pos, size, 0x00ff00ff);

	// 		printf("voxelIndex: %i, points: %i \n", subgrid.voxelIndex, subgrid.numPoints);
			
	// 	}
	// });

	distribute(local_root, points_unsorted, points_sorted, box, boxSize, gridSize, depth, countgrids, nodegrids);

	// if(grid.thread_rank() == 0){
	// 	printf("allocator->offset: %llu \n ", allocator->offset);

	// 	// now revert the allocations taken during this function
	// 	allocator->offset = original_allocator_offset;
	// }

	allocator->offset = original_allocator_offset;

	grid.sync();
	local_root->numPoints = 0;
	local_root->points = nullptr;
	grid.sync();
}

void main_split(
	Allocator& _allocator, 
	Box3 box, 
	Point* input_points, 
	int numPoints,
	void** nnodes,
	uint32_t* nnum_nodes,
	void** ssorted,
	Points* _points,
	Lines* _lines
){
	auto grid  = cg::this_grid();
	auto block = cg::this_thread_block();

	lines = _lines;
	points = _points;
	allocator = &_allocator;

	dbg = allocator->alloc<uint32_t*>(4);
	*dbg = 0;

	uint32_t& numNodes = *allocator->alloc<uint32_t*>(4, "node counter");
	Node* nodes        = allocator->alloc<Node*>(sizeof(Node) * MAX_NODES, "nodes");
	Point* sorted      = allocator->alloc<Point*>(sizeof(Point) * numPoints, "sorted");

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
	split_node(root, input_points, sorted, depth, nodes, numNodes, lines);
	
	grid.sync();

	*nnodes = (void*)nodes;
	*ssorted = (void*)sorted;
	*nnum_nodes = numNodes;

	grid.sync();

	if(PRINT_STATS)
	if(isFirstThread())
	{
		printf("split_countsort_blockwise done\n");
		printf("allocated memory: %i MB \n", int(allocator->offset / (1024 * 1024)));
		printf("#nodes:           %i \n", numNodes);
		printf("#subGrids:      %i \n", *numSubGrids);
	}
}


};
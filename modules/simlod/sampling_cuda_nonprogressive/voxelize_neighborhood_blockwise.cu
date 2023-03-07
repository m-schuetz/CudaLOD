

namespace voxelize_neighborhood_blockwise{

#include <cooperative_groups.h>
#include "lib.h.cu"
#include "methods_common.h.cu"

namespace cg = cooperative_groups;

constexpr int clearGridSize = 8;

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

// constexpr int MODE_CENTRAL = 0;
// constexpr int MODE_ADJACENT = 1;

void voxelizePrimitives_central(
	Point* points, 
	uint32_t numPoints, 
	Node* node, 
	int gridSize,
	uint32_t* voxelGrid,
	vec3 boxSize, 
	uint32_t& sh_numAccepted, 
	uint32_t* accepted,
	uint32_t* sh_clearGrid
){
	auto block = cg::this_thread_block();

	float fGridSize = gridSize;

	int numIterations = numPoints / block.num_threads() + 1;
	for(int it = 0; it < numIterations; it++){
		int pointIndex = block.num_threads() * it + block.thread_rank();

		if(pointIndex >= numPoints) continue;

		Point point = points[pointIndex];

		// project to node's 128³ sample grid
		float fx = fGridSize * (point.x - node->min.x) / boxSize.x;
		float fy = fGridSize * (point.y - node->min.y) / boxSize.y;
		float fz = fGridSize * (point.z - node->min.z) / boxSize.z;

		vec3 pos = {fx, fy, fz};

		{

			vec3 samplePos = vec3(
				floor(fx) + 0.5f,
				floor(fy) + 0.5f,
				floor(fz) + 0.5f
			);

			float dx = (pos.x - samplePos.x);
			float dy = (pos.y - samplePos.y);
			float dz = (pos.z - samplePos.z);
			float ll = (dx * dx + dy * dy + dz * dz);
			float w = 0.0f;

			float l = sqrt(ll);

			if(ll < 1.0f){
				// exponential filter
				// w = __expf(-ll * 0.5f);
				// w = clamp(w, 0.0f, 1.0f);
				
				// linear filter
				w = 1.0 - l;
			}else{
				w = 0.0;
			}

			if(w > 0.0f){

				uint64_t W = clamp(100.0f * w, 1.0f, 100.0f);

				uint32_t ix = clamp(samplePos.x, 0.0f, fGridSize - 1.0f);
				uint32_t iy = clamp(samplePos.y, 0.0f, fGridSize - 1.0f);
				uint32_t iz = clamp(samplePos.z, 0.0f, fGridSize - 1.0f);

				uint32_t voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;


				// if(ox == 0.0f && oy == 0.0f && oz == 0.0f){
				// 	uint32_t res = atomicOr(&voxelGrid[4 * voxelIndex + 3], (1u << 31u));
				// 	bool isNewlyOccupied = (res & (1u << 31u)) == 0;
					
				// 	if (isNewlyOccupied){
				// 		uint32_t acceptedIndex = atomicAdd(&sh_numAccepted, 1);
				// 		accepted[acceptedIndex] = voxelIndex;
				// 	}
				// }

				uint64_t* cell = (uint64_t*)&voxelGrid[4 * voxelIndex + 0];

				uint8_t* rgba = (uint8_t*)&point.color;
				uint64_t R = W * rgba[0];
				uint64_t G = W * rgba[1];
				uint64_t B = W * rgba[2];
				atomicAdd(cell + 0, uint64_t(R | (G << 32)));
				uint64_t old = atomicAdd(cell + 1, uint64_t(B | (W << 32)));


				bool isNewlyOccupied = (old >> 32) == 0;
				if (isNewlyOccupied){
					uint32_t acceptedIndex = atomicAdd(&sh_numAccepted, 1);
					accepted[acceptedIndex] = voxelIndex;
				}
			}
		}
	}
}

void voxelizePrimitives_neighbors(
	Point* points, 
	uint32_t numPoints, 
	Node* node, 
	int gridSize,
	uint32_t* voxelGrid,
	vec3 boxSize, 
	uint32_t& sh_numAccepted, 
	uint32_t* accepted,
	uint32_t* sh_clearGrid
){
	auto block = cg::this_thread_block();

	float fGridSize = gridSize;

	int numIterations = numPoints / block.num_threads() + 1;
	for(int it = 0; it < numIterations; it++){
		int pointIndex = block.num_threads() * it + block.thread_rank();

		if(pointIndex >= numPoints) continue;

		Point point = points[pointIndex];

		// project to node's 128³ sample grid
		float fx = fGridSize * (point.x - node->min.x) / boxSize.x;
		float fy = fGridSize * (point.y - node->min.y) / boxSize.y;
		float fz = fGridSize * (point.z - node->min.z) / boxSize.z;

		vec3 pos = {fx, fy, fz};

		for(float oz : {-1.0f, 0.0f, 1.0f})
		for(float oy : {-1.0f, 0.0f, 1.0f})
		for(float ox : {-1.0f, 0.0f, 1.0f})
		{

			vec3 samplePos = vec3(
				floor(fx + ox) + 0.5f,
				floor(fy + oy) + 0.5f,
				floor(fz + oz) + 0.5f
			);

			float dx = (pos.x - samplePos.x);
			float dy = (pos.y - samplePos.y);
			float dz = (pos.z - samplePos.z);
			float ll = (dx * dx + dy * dy + dz * dz);
			float w = 0.0f;

			float l = sqrt(ll);

			if(ll < 1.0f){
				// exponential filter
				// w = __expf(-ll * 0.5f);
				// w = clamp(w, 0.0f, 1.0f);
				
				// linear filter
				w = 1.0 - l;
			}else{
				w = 0.0;
			}

			if(w > 0.0f){

				uint64_t W = clamp(100.0f * w, 1.0f, 100.0f);

				uint32_t ix = clamp(samplePos.x, 0.0f, fGridSize - 1.0f);
				uint32_t iy = clamp(samplePos.y, 0.0f, fGridSize - 1.0f);
				uint32_t iz = clamp(samplePos.z, 0.0f, fGridSize - 1.0f);

				uint32_t voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

				bool isCenter = ox == 0.0f && oy == 0.0f && oz == 0.0f;
				bool isNeighbor = !isCenter;
				if(isNeighbor){
					uint64_t* cell = (uint64_t*)&voxelGrid[4 * voxelIndex + 0];

					uint32_t currentW = voxelGrid[4 * voxelIndex + 3];

					if(currentW > 0){
						uint8_t* rgba = (uint8_t*)&point.color;
						uint64_t R = W * rgba[0];
						uint64_t G = W * rgba[1];
						uint64_t B = W * rgba[2];
						atomicAdd(cell + 0, uint64_t(R | (G << 32)));
						atomicAdd(cell + 1, uint64_t(B | (W << 32)));
					}
				}

				
			}
		}
	}
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
	uint32_t& nodeCounter          = *allocator.alloc<uint32_t*>(4);

	if(isFirstThread()){
		totalVoxelBufferSize = 0;
		nodeVoxelBufferSize = 0;
		clearCounter = 0;
		nodeCounter = 0;
	}

	grid.sync();
	
	int gridSize               = VOXEL_GRID_SIZE;
	// float fGridSize            = gridSize;
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

	allocator.alloc<uint32_t*>(10000 * 16);
	uint64_t& globalAllocatorOffset = *allocator.alloc<uint64_t*>(8);
	allocator.alloc<uint32_t*>(10000 * 16);

	__shared__ uint32_t sh_workIndex;
	__shared__ uint32_t sh_numAccepted;
	__shared__ uint32_t sh_clearGrid[clearGridSize * clearGridSize * clearGridSize];

	// initially clear all voxel grids
	clearBuffer(voxelGrids, 0, grid.num_blocks() * voxelGridByteSize, 0);

	grid.sync();

	if(isFirstThread()){
		globalAllocatorOffset = allocator.offset;

		// printf("allocator.offset: ");
		// printNumber(allocator.offset, 10);
		// printf("\n");
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

			{ // this assumes a workgroup size of 256!!!
				if(block.num_threads() != 256)
				if(block.thread_rank() == 0)
				{
					printf("error, expecting a workgroup size of 256");
				}

				sh_clearGrid[2 * block.thread_rank() + 0] = 0;
				sh_clearGrid[2 * block.thread_rank() + 1] = 0;
			}

			block.sync();

			// first, central projection
			for(int childIndex = 0; childIndex < 8; childIndex++){
				Node* child = node->children[childIndex];

				if(child == nullptr) continue;

				block.sync();

				// POINTS
				voxelizePrimitives_central(
					child->points, child->numPoints, node, 
					gridSize, voxelGrid, boxSize, sh_numAccepted, accepted, sh_clearGrid);

				block.sync();

				// VOXELS
				voxelizePrimitives_central(
					child->voxels, child->numVoxels, node, 
					gridSize, voxelGrid, boxSize, sh_numAccepted, accepted, sh_clearGrid);

				block.sync();
			}

			block.sync();

			// then, neighbor projection.
			// neighbors only modify cells that were
			// occupied by the central projection,
			// which allows us to quickly clear only relevant cells
			for(int childIndex = 0; childIndex < 8; childIndex++){
				Node* child = node->children[childIndex];

				if(child == nullptr) continue;

				block.sync();

				// POINTS
				voxelizePrimitives_neighbors(
					child->points, child->numPoints, node, 
					gridSize, voxelGrid, boxSize, sh_numAccepted, accepted, sh_clearGrid);

				block.sync();

				// VOXELS
				voxelizePrimitives_neighbors(
					child->voxels, child->numVoxels, node, 
					gridSize, voxelGrid, boxSize, sh_numAccepted, accepted, sh_clearGrid);

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

			// { // DEBUG: clear voxel buffer first (should not be necessary)
			// 	block.sync();

			// 	int numIterations = sh_numAccepted / block.num_threads() + 1;
			// 	for(int it = 0; it <= numIterations; it++){
			// 		int index = block.num_threads() * it + block.thread_rank();

			// 		if(index >= sh_numAccepted) continue;

			// 		Point point;
			// 		point.x = 0;
			// 		point.y = 0;
			// 		point.z = 0;
			// 		point.color = 0;
			// 		node->voxels[index] = point;

			// 	}
			// }

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
				uint32_t W = voxelGrid[4 * voxelIndex + 3];

				// bool occupied = (W & (1u << 31u)) != 0u;
				// W = W & 0b0111111'11111111'11111111'11111111;

				uint32_t color;
				uint8_t* rgba = (uint8_t*)&color;
				rgba[0] = R / W;
				rgba[1] = G / W;
				rgba[2] = B / W;

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

				bool outsideX = pos.x < node->min.x || pos.x >= node->max.x;
				bool outsideY = pos.y < node->min.y || pos.y >= node->max.y;
				bool outsideZ = pos.z < node->min.z || pos.z >= node->max.z;
				if(outsideX || outsideY || outsideZ){
					// printf("out of bounds \n");
				}

				// if(node->level == 3 && node->voxelIndex == 99){
				// 	if(voxel.y > 2.32621408f - 0.01f && voxel.y < 2.32621408f + 0.01f)
				// 	if(voxel.z > 0.841976047f - 0.01f && voxel.z < 0.841976047f + 0.01f)
				// 	// if(index < 10)
				// 	{
				// 		printf("xyz: %f, %f, %f \n", voxel.x, voxel.y, voxel.z);
				// 	}
				// }

				// if(voxel.x == 0.0f && voxel.y == 0.0f && voxel.z == 0.0f){
				// 	printf("xyz: %f, %f, %f \n", voxel.x, voxel.y, voxel.z);
				// }
				

				node->voxels[index] = voxel;

				// Since only cells that contain points are affected,
				// we can directly clear the cell now.
				// (neighbors dont modify cells without actual geometry)
				voxelGrid[4 * voxelIndex + 0] = 0;
				voxelGrid[4 * voxelIndex + 1] = 0;
				voxelGrid[4 * voxelIndex + 2] = 0;
				voxelGrid[4 * voxelIndex + 3] = 0;
			}

			if(block.thread_rank() == 0){
				node->dbg = atomicAdd(&nodeCounter, 1);
			}
			
			block.sync();
		}
	}

	grid.sync();

	// update allocator.offset for subsequent allocations in "kernel.cu"
	allocator.offset = globalAllocatorOffset;

	// PRINT("smallVolumeNodeCounter: %i \n", smallVolumeNodeCounter);
	// PRINT("smallVolumePointCounter: %i k \n", (smallVolumePointCounter / 1000) );
}

};
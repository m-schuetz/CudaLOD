

namespace singlecell{

#include <cooperative_groups.h>
#include "lib.h.cu"
#include "methods_common.h.cu"

namespace cg = cooperative_groups;

void voxelizePrimitives(Node* node, vec3 boxSize, uint32_t num, 
	Point* primitives, uint32_t offset, uint32_t gridSize, float fGridSize, 
	uint32_t* voxelGrid, uint32_t* voxelBufferSize, Point* voxelBuffer,
	int dbg = 0, int dbg2 = 0
){

	processRange(0, num, [&](int pointIndex){

		Point point = primitives[offset + pointIndex];

		float fx = fGridSize * (point.x - node->min.x) / boxSize.x;
		float fy = fGridSize * (point.y - node->min.y) / boxSize.y;
		float fz = fGridSize * (point.z - node->min.z) / boxSize.z;

		{ // AVERAGE


			uint32_t ix = clamp(fx, 0.0f, fGridSize - 1.0f);
			uint32_t iy = clamp(fy, 0.0f, fGridSize - 1.0f);
			uint32_t iz = clamp(fz, 0.0f, fGridSize - 1.0f);

			uint32_t voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

			uint32_t R = ((point.color >>  0) & 0xff);
			uint32_t G = ((point.color >>  8) & 0xff);
			uint32_t B = ((point.color >> 16) & 0xff);
			int A = 1;

			uint64_t* voxelGrid64 = (uint64_t*)voxelGrid;

			uint64_t RG = (uint64_t(G) << 32ull) | R;
			uint64_t BA = (uint64_t(A) << 32ull) | B;
			atomicAdd(&voxelGrid64[2 * voxelIndex + 0], RG);
			auto old = atomicAdd(&voxelGrid64[2 * voxelIndex + 1], BA);

			if(old == 0){
				uint32_t rindex = atomicAdd(voxelBufferSize, 1);
				voxelBuffer[rindex].color = voxelIndex;
			}
		}


		// { // WEIGHTED AVERAGE
		// 	vec3 pos = {fx, fy, fz};
		// 	vec3 samplePos = vec3(
		// 		floor(fx) + 0.5f,
		// 		floor(fy) + 0.5f,
		// 		floor(fz) + 0.5f
		// 	);

		// 	float dx = (pos.x - samplePos.x);
		// 	float dy = (pos.y - samplePos.y);
		// 	float dz = (pos.z - samplePos.z);
		// 	float ll = (dx * dx + dy * dy + dz * dz);
		// 	float w = 0.0f;

		// 	if(ll < 1.0f){
		// 		w = __expf(-ll * 3.5f);
		// 		w = clamp(w, 0.0f, 1.0f);
		// 	}

		// 	if(w > 0.0f)
		// 	{

		// 		int W = clamp(100.0f * w, 1.0f, 100.0f);
		// 		// int W = 1;

		// 		uint32_t ix = clamp(samplePos.x, 0.0f, fGridSize - 1.0f);
		// 		uint32_t iy = clamp(samplePos.y, 0.0f, fGridSize - 1.0f);
		// 		uint32_t iz = clamp(samplePos.z, 0.0f, fGridSize - 1.0f);

		// 		uint32_t voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

		// 		uint32_t R = W * ((point.color >>  0) & 0xff);
		// 		uint32_t G = W * ((point.color >>  8) & 0xff);
		// 		uint32_t B = W * ((point.color >> 16) & 0xff);

		// 		atomicAdd((uint64_t*)&voxelGrid[4 * voxelIndex + 0], (((uint64_t)G) << 32) | R);
		// 		auto old = atomicAdd((uint64_t*)&voxelGrid[4 * voxelIndex + 2], (((uint64_t)W) << 32) | B);

		// 		if(old == 0){
		// 			uint32_t rindex = atomicAdd(voxelBufferSize, 1);
		// 			voxelBuffer[rindex].color = voxelIndex;
		// 		}
		// 	}
		// }
		
	});
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

	{
		int gridSize        = VOXEL_GRID_SIZE;
		float fGridSize     = gridSize;
		int numCells        = gridSize * gridSize * gridSize;
		uint32_t* voxelGrid = allocator.alloc<uint32_t*>(sizeof(uint32_t) * 4 * numCells, "voxel sampling grid RG");
		// uint32_t* voxelGrid_RG = allocator.alloc<uint32_t*>(sizeof(uint32_t) * 2 * numCells, "voxel sampling grid RG");
		// uint32_t* voxelGrid_BA = allocator.alloc<uint32_t*>(sizeof(uint32_t) * 2 * numCells, "voxel sampling grid BA");

		clearBuffer(voxelGrid, 0, sizeof(uint32_t) * 4 * numCells, 0);
		// clearBuffer(voxelGrid_RG, 0, sizeof(uint32_t) * 2 * numCells, 0);
		// clearBuffer(voxelGrid_BA, 0, sizeof(uint32_t) * 2 * numCells, 0);

		for(int abc = 0; abc < 12; abc++){

			grid.sync();

			auto t_create_workload = Timer::start("create workload");
			{ // FIND DEEPEST EMPTY INNER NODES (TO FILL WITH VOXELS)
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
			t_create_workload.stop();


			if(workloadSize == 0){
				// we're done!
				break;
			}

			{ // CREATE VOXEL LOD FOR DEEPEST EMPTY INNER NODES

				for(int nodeIndex = 0; nodeIndex < workloadSize; nodeIndex++){
					
					Node* node = workload[nodeIndex];
					Point* voxelBuffer = reinterpret_cast<Point*>(allocator.buffer + allocator.offset + sizeof(Point) * totalVoxelBufferSize);

					grid.sync();

					vec3 boxSize = node->max - node->min;
					// clearBuffer(voxelGrid, 0, sizeof(uint32_t) * 4 * numCells, 0);
					// clearBuffer(voxelGrid_RG, 0, sizeof(uint32_t) * 2 * numCells, 0);
					// clearBuffer(voxelGrid_BA, 0, sizeof(uint32_t) * 2 * numCells, 0);
					// if(grid.thread_rank() == 0){
					// 	atomicAdd(&clearCounter, 1);
					// }

					grid.sync();

					// auto t_voxelize = Timer::start("voxelize");
					for(int childIndex = 0; childIndex < 8; childIndex++){
						Node* child = node->children[childIndex];
						if(child == nullptr) continue;

						grid.sync();

						// VOXELIZE POINTS
						voxelizePrimitives(node, boxSize, child->numPoints, 
							child->points, 0, gridSize, fGridSize, 
							voxelGrid, &nodeVoxelBufferSize, voxelBuffer, 0);
						
						if(abc == 0) continue;

						grid.sync();

						// VOXELIZE VOXELS
						voxelizePrimitives(node, boxSize, child->numVoxels, 
							child->voxels, 0, gridSize, fGridSize,
							voxelGrid, &nodeVoxelBufferSize, voxelBuffer,
							abc, nodeIndex
						);
					}
					// t_voxelize.stop();

					grid.sync();

					// EXTRACT VOXELS
					// auto t_extract = Timer::start("extract");
					processRange(0, nodeVoxelBufferSize, [&](int rindex){
						uint32_t index = voxelBuffer[rindex].color;
						int ix = index % gridSize;
						int iy = (index % (gridSize * gridSize)) / gridSize;
						int iz = index / (gridSize * gridSize);

						uint32_t W = voxelGrid[4 * index + 3];

						if(W > 0){
							float x = (float(ix) + 0.5f) * boxSize.x / float(gridSize);
							float y = (float(iy) + 0.5f) * boxSize.y / float(gridSize);
							float z = (float(iz) + 0.5f) * boxSize.z / float(gridSize);
							float cubeSize = boxSize.x / float(gridSize);

							vec3 pos = {x, y, z};
							pos = pos + node->min;
							vec3 size = {cubeSize, cubeSize, cubeSize};
							uint32_t color = 0x000000ff;

							uint32_t R = float(voxelGrid[4 * index + 0]) / float(W);
							uint32_t G = float(voxelGrid[4 * index + 1]) / float(W);
							uint32_t B = float(voxelGrid[4 * index + 2]) / float(W);

							R = R & 0xff;
							G = G & 0xff;
							B = B & 0xff;

							int isize = 300.0f * size.x;
							color = R | (G << 8) | (B << 16) | (isize << 24);
							// color = 0x000000ff;

							Point voxel;
							voxel.x = pos.x;
							voxel.y = pos.y;
							voxel.z = pos.z;
							voxel.color = color;

							voxelBuffer[rindex] = voxel;

							// since the singlecell method only projects to cells with actual geometry,
							// we can clear the voxelGrid directly here during extraction
							voxelGrid[4 * index + 0] = 0;
							voxelGrid[4 * index + 1] = 0;
							voxelGrid[4 * index + 2] = 0;
							voxelGrid[4 * index + 3] = 0;
						}
					});
					// t_extract.stop();

					grid.sync();


					grid.sync();
					
					if(isFirstThread()){
						node->voxels = voxelBuffer;
						node->numVoxels = nodeVoxelBufferSize;

						atomicAdd(&totalVoxelBufferSize, nodeVoxelBufferSize);
						atomicExch(&nodeVoxelBufferSize, 0);
					}
					
					grid.sync();
				}
			}
		}

		grid.sync();

		// post-allocation of generated voxels
		// we dynamically placed voxels starting from previous allocator.offset 
		// now that we've done that, we know how much memory we needed and update the allocator
		allocator.alloc<Point*>(sizeof(Point) * totalVoxelBufferSize, "all voxels");
	}

	grid.sync();

	

}

};
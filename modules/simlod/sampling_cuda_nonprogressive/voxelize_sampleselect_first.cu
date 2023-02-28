
namespace sampleselect_first{

#include <cooperative_groups.h>
#include "lib.h.cu"
#include "methods_common.h.cu"

namespace cg = cooperative_groups;

void voxelizePrimitives(Node* node, vec3 boxSize, uint32_t num, 
	Point* points, uint32_t gridSize, float fGridSize, 
	uint64_t* voxelGrid, uint32_t* voxelBufferSize, Point* voxelBuffer
){

	processRange(0, num, [&](int pointIndex){

		Point point = points[pointIndex];

		float fx = fGridSize * (point.x - node->min.x) / boxSize.x;
		float fy = fGridSize * (point.y - node->min.y) / boxSize.y;
		float fz = fGridSize * (point.z - node->min.z) / boxSize.z;

		uint32_t ix = clamp(fx, 0.0f, fGridSize - 1.0f);
		uint32_t iy = clamp(fy, 0.0f, fGridSize - 1.0f);
		uint32_t iz = clamp(fz, 0.0f, fGridSize - 1.0f);

		uint32_t voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

		auto old = atomicCAS(&voxelGrid[voxelIndex], 0, (uint64_t)&points[pointIndex]);

		if(old == 0){
			uint32_t rindex = atomicAdd(voxelBufferSize, 1);
			voxelBuffer[rindex].color = voxelIndex;
		}


		// { 
		// 	auto warp = cg::coalesced_threads();
		// 	auto group = cg::labeled_partition(warp, voxelIndex);

		// 	if(group.thread_rank() == 0){
		// 		auto old = atomicCAS(&voxelGrid[voxelIndex], 0, (uint64_t)&points[pointIndex]);

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
		uint64_t* voxelGrid = allocator.alloc<uint64_t*>(sizeof(uint64_t) * numCells, "voxel sampling grid RG");

		clearBuffer(voxelGrid, 0, sizeof(uint64_t) * numCells, 0);

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

					grid.sync();

					// auto t_voxelize = Timer::start("voxelize");
					for(int childIndex = 0; childIndex < 8; childIndex++){
						Node* child = node->children[childIndex];
						if(child == nullptr) continue;

						grid.sync();

						// VOXELIZE POINTS
						voxelizePrimitives(node, boxSize, child->numPoints, 
							child->points, gridSize, fGridSize, 
							voxelGrid, &nodeVoxelBufferSize, voxelBuffer);

						grid.sync();

						// VOXELIZE VOXELS
						voxelizePrimitives(node, boxSize, child->numVoxels, 
							child->voxels, gridSize, fGridSize,
							voxelGrid, &nodeVoxelBufferSize, voxelBuffer);
					}
					// t_voxelize.stop();

					grid.sync();

					// EXTRACT VOXELS
					// auto t_extract = Timer::start("extract");
					processRange(0, nodeVoxelBufferSize, [&](int rindex){
						
						uint32_t index = voxelBuffer[rindex].color;
						Point* ptr = reinterpret_cast<Point*>(voxelGrid[index]);

						if(ptr != nullptr){
							Point point = *ptr;

							voxelBuffer[rindex] = point;
							voxelGrid[index] = 0;
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

	// if(false)
	if(isFirstThread())
	{ // DEBUG

		printf("#voxels:                 %8i \n", totalVoxelBufferSize);
		printf("allocated memory:        "); //%i MB \n", allocator.offset / (1024 * 1024));
		printNumber(allocator.offset, 13);
		printf("\n");
		printf("clear counter: %i \n", clearCounter);
	}

}

};
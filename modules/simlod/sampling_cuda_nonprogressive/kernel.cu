#define VERBOSE false

#include <cooperative_groups.h>
#include "lib.h.cu"
#include "helper_math.h"
#include "common.h"
#include <curand_kernel.h>

#include "methods_common.h.cu"

#include "voxelize_singlecell.cu"
#include "voxelize_neighborhood.cu"
#include "voxelize_sampleselect_first.cu"
#include "voxelize_sampleselect_random.cu"
#include "voxelize_sampleselect_first_blockwise.cu"
#include "voxelize_sampleselect_random_blockwise.cu"
#include "voxelize_sampleselect_random_blockwise_globalmem.cu"
// #include "voxelize_sampleselect_random_blockwise_nonsparse.cu"
#include "voxelize_singlecell_blockwise.cu"
#include "voxelize_neighborhood_blockwise.cu"
#include "voxelize_sampleselect_central_blockwise.cu"
   
namespace cg = cooperative_groups;
 
int dbg;                                  
 
struct Buffer{
	int offset;
	int size;
	uint8_t* ptr;
};
 
struct {
	unsigned int* buffer;
	Point* input_points;
	State state;
	Allocator* allocator;
	Lines* lines;
} globals;

#include "split_countsort.h.cu"
#include "split_countsort_blockwise.h.cu"
#include "split_hashmap.h.cu"
// #include "voxelsampling.h.cu"


constexpr int numBins = 100;
struct Histogram{

	int counters[numBins];
	int val_min;
	int val_max;

	void init(int min, int max){
		this->val_min = min;
		this->val_max = max;

		for(int i = 0; i < numBins; i++){
			counters[i] = 0;
		}
	}

	void add(int value){
		int binID = float(numBins) * float(value) / float(val_max);
		binID = min(binID, numBins - 1);
		atomicAdd(&counters[binID], 1);
	}

	void print(const char* label, int lines){

		int actualMax = 0;
		for(int i = 0; i < numBins; i++){
			actualMax = max(actualMax, counters[i]);
		}

		// printf("== HISTOGRAM - POINTS ==\n");
		printf("== %s ============================================\n", label);
		printf("| min: %6i, max: %6i, #bins: %i \n", val_min, val_max, numBins);
		printf("| \n");

		for(int line = lines - 1; line >= 0; line--){
			printf("|");
			for(int i = 0; i < numBins; i++){

				uint32_t binValue = counters[i];

				float u = float(binValue) / float(actualMax);
				float t = float(line) / float(lines);

				if(u >= t && u > 0.0f){
					printf(" #### ");
				}else{
					printf("      ");
				}

			}
			printf("\n");
		}

		// print number of items per bin
		printf("|");
		for(int i = 0; i < numBins; i++){

			if(counters[i] > 0){
				printf("%4i, ", counters[i]);
			}else{
				printf("      ");
			}
		}
		printf("\n");

		// print bin range
		printf("|");
		for(int i = 0; i < numBins; i++){
			float u = float(i + 1)  / float(numBins);
			int rangeMax = u * float(val_max);
			printf("%3ik, ", rangeMax / 1000);
		}
		printf("\n");

		printf("=================================================================\n");
	}

};

extern "C" __global__
void kernel2(
	State state,
	unsigned int* buffer,
	Results* results,
	Point* input_points,
	void** nodes,
	uint32_t* num_nodes,
	void** sorted,
	uint64_t* alloc_offset,
	void** _points,
	void** _lines
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	if(grid.thread_rank() == 0){
		// printf("==============================\n");
		// printf("=== STARTING KERNEL 2! \n");
		// printf("==============================\n");

		Timer::instance = new Timer();
		Timer::instance->t_start = clock64();
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(Timer::instance->t_start_nano));
	}

	Allocator allocator(buffer, 0);

	Lines* lines = allocator.alloc<Lines*>(sizeof(Lines));
	Points* points = allocator.alloc<Points*>(sizeof(Points));
	lines->count = 0;
	points->count = 0;
	*_lines = (void*)lines;
	*_points = (void*)points;

	grid.sync();
	// if(grid.thread_rank() == 0){
	// 	vec3 start = {0.0f, 0.0f, 0.0f};
	// 	vec3 end = {100.0f, 100.0f, 100.0f};
	// 	drawLine(lines, start, end, 0x0000ff00);
	// 	drawBoundingBox(lines, {0.0f, 0.0f, 0.0f}, {100.0f, 100.0f, 100.0f}, 0x000000ff);
	// }
	
	// GLOBAL STATE
	globals.buffer        = buffer;
	globals.input_points  = input_points;
	globals.state         = state;
	globals.allocator     = &allocator;
	globals.lines         = lines;

	Box3 box;    
	{
		box.min = {state.metadata.min_x, state.metadata.min_y, state.metadata.min_z};
		box.max = {state.metadata.max_x, state.metadata.max_y, state.metadata.max_z};
		float cubeSize = box.size().longestAxis();
		box.max = box.min + cubeSize;
	} 

	grid.sync();

	{// PICK SPLIT METHOD
		// whole grid processes one node at a time
		// split_countsort::main_split(allocator, box, input_points, state.metadata.numPoints, nodes, num_nodes, sorted);

		// One workgroup per node on second split. 
		split_countsort_blockwise::main_split(allocator, box, input_points, state.metadata.numPoints, nodes, num_nodes, sorted, points, lines);
	}

	// prototyping, dont use
	// split_hashmap::main_split(allocator, box, input_points, state.metadata.numPoints, nodes, num_nodes, sorted);

	if(grid.thread_rank() == 0){
		Timer::instance->t_end = clock64();
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(Timer::instance->t_end_nano));
	}

	grid.sync();

	// Timer::print();

	results->allocatedMemory_splitting = allocator.offset;

	*alloc_offset = allocator.offset;
}
                           
extern "C" __global__
void kernel3(
	State state,
	unsigned int* buffer,
	Results* results,
	Point* input_points,
	void** nodes,
	uint32_t* num_nodes,
	void** sorted,
	uint64_t* alloc_offset,
	void** _points,
	void** _lines
){ 
	// return; 
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	if(grid.thread_rank() == 0){
		// printf("==============================\n");
		// printf("=== STARTING KERNEL 3! \n");
		// printf("==============================\n");

		Timer::instance = new Timer();
		Timer::instance->t_start = clock64();
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(Timer::instance->t_start_nano));
	}
	
	Allocator allocator(buffer, *alloc_offset);

	// Lines* lines = (Lines*)*_lines;
	// Points* points = (Points*)*_points;

	grid.sync();
	
	Box3 box; 
	box.min = {state.metadata.min_x, state.metadata.min_y, state.metadata.min_z};
	box.max = {state.metadata.max_x, state.metadata.max_y, state.metadata.max_z};
	float cubeSize = box.size().longestAxis();
	box.max = box.min + cubeSize;

	// note that this dynamic branching has the potentiall issue that 
	// FIRST_COME reserves shared memory even if another method is selected
	if(state.strategy == SamplingStrategy::FIRST_COME){
		sampleselect_first_blockwise::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);
	}else if(state.strategy == SamplingStrategy::RANDOM){
		sampleselect_random_blockwise_globalmem::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);
	}else if(state.strategy == SamplingStrategy::AVERAGE_SINGLECELL){
		voxelize_singlecell_blockwise::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);
	}else if(state.strategy == SamplingStrategy::WEIGHTED_NEIGHBORHOOD){
		voxelize_neighborhood_blockwise::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);
	}
	 
	{ // PICK VOXELIZATION METHOD
		// sampleselect_random_blockwise_globalmem::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);
		
		// FIRST-COME
		// sampleselect_first_blockwise::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);

		// RANDOM
		// sampleselect_random::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);
		// sampleselect_random_blockwise::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);

		// AVERAGE IN CELL
		// singlecell::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);
		// voxelize_singlecell_blockwise::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);

		// WEIGHTED AVERAGE IN NEIGHBORHOOD
		// voxelize_neighborhood_blockwise::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);
		// neighborhood::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);
 
	}

	// MISC, just for prototyping
	// sampleselect_first_blockwise::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);
	// sampleselect_central_blockwise::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);
	// sampleselect_first::main_voxelize(allocator, box, state.metadata.numPoints, *nodes, *num_nodes, *sorted);
    
	if(grid.thread_rank() == 0){
		Timer::instance->t_end = clock64();
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(Timer::instance->t_end_nano));
	}

	grid.sync();


	// if(PRINT_STATS)
	{// COMPUTE STATS
		uint32_t numPoints = state.metadata.numPoints;
		uint32_t numNodes = *num_nodes;
		void* nnnodes = *nodes;
		Node* nodelist = (Node*)nnnodes;

		uint32_t& stat_num_leaves  = *allocator.alloc<uint32_t*>(4);
		uint32_t& stat_num_inner   = *allocator.alloc<uint32_t*>(4);
		uint64_t& stat_sum_leaves  = *allocator.alloc<uint64_t*>(8);
		uint64_t& stat_sum_inner   = *allocator.alloc<uint64_t*>(8);
		uint64_t& stat_max_leaves  = *allocator.alloc<uint64_t*>(8);
		uint64_t& stat_max_inner   = *allocator.alloc<uint64_t*>(8);
		uint64_t& stat_min_leaves  = *allocator.alloc<uint64_t*>(8);
		uint64_t& stat_min_inner   = *allocator.alloc<uint64_t*>(8);
		
		uint64_t* pointsAtLevel    = allocator.alloc<uint64_t*>(50 * 8);
		uint64_t* voxelsAtLevel    = allocator.alloc<uint64_t*>(50 * 8);
		uint64_t* innerAtLevel     = allocator.alloc<uint64_t*>(50 * 8);
		uint64_t* leavesAtLevel    = allocator.alloc<uint64_t*>(50 * 8);

		// clearBuffer(samplesAtLevel, 0, 20 * 8, 0);
		clearBuffer(pointsAtLevel, 0, 50 * 8, 0);
		clearBuffer(voxelsAtLevel, 0, 50 * 8, 0);
		clearBuffer(innerAtLevel , 0, 50 * 8, 0);
		clearBuffer(leavesAtLevel, 0, 50 * 8, 0);

		Histogram* hist_points = allocator.alloc<Histogram*>(sizeof(Histogram));
		Histogram* hist_voxels = allocator.alloc<Histogram*>(sizeof(Histogram));

		if(grid.thread_rank() == 0){
			hist_points->init(0, 100'000);
			hist_voxels->init(0, 100'000);

		}

		if(isFirstThread()){
			stat_num_leaves  = 0;
			stat_num_inner   = 0;
			stat_sum_leaves  = 0;
			stat_sum_inner   = 0;
			stat_max_leaves  = 0;
			stat_max_inner   = 0;
			stat_min_leaves  = 1'000'000'000;
			stat_min_inner   = 1'000'000'000;
		}
 
		grid.sync();

		processRange(0, numNodes, [&](int nodeIndex){
			Node* node = &nodelist[nodeIndex];

			if(node->isLeaf())
			// if(node->numPoints > 0)
			{
				atomicAdd(&stat_num_leaves, 1);
				atomicAdd(&stat_sum_leaves, node->numPoints);
				atomicMax(&stat_max_leaves, node->numPoints);
				atomicMin(&stat_min_leaves, node->numPoints);

				hist_points->add(node->numPoints);

			}else{
				atomicAdd(&stat_num_inner, 1);
				atomicAdd(&stat_sum_inner, node->numVoxels);
				atomicMax(&stat_max_inner, node->numVoxels);
				atomicMin(&stat_min_inner, node->numVoxels);

				hist_voxels->add(node->numVoxels);
			}

			if(node->level >= 0 && node->level < 20){
				atomicAdd(&pointsAtLevel[node->level], node->numPoints);
				atomicAdd(&voxelsAtLevel[node->level], node->numVoxels);
				
				if(node->numPoints > 0){
					atomicAdd(&leavesAtLevel[node->level], 1);
				}else if(node->numVoxels > 0){
					atomicAdd(&innerAtLevel[node->level], 1);
				}else{
					assert(false);
				}
			}else{
				printf("wtf\n");
			}

		});

		grid.sync();

		if(grid.thread_rank() == 0){
			results->strategy   = state.strategy;
			results->points     = numPoints;
			results->voxels     = stat_sum_inner;
			results->nodes      = numNodes;
			results->innerNodes = stat_num_inner;
			results->leafNodes  = stat_num_leaves;
			results->minPoints  = stat_min_leaves;
			results->maxPoints  = stat_max_leaves;
			results->avgPoints  = stat_sum_leaves / stat_num_leaves;
			results->minVoxels  = stat_min_inner;
			results->maxVoxels  = stat_max_inner;
			results->avgVoxels  = stat_sum_inner / stat_num_inner;

			results->histogram_maxval = hist_points->val_max;
			
			for(int i = 0; i < 100; i++){
				results->histogram_points[i] = hist_points->counters[i];
				results->histogram_voxels[i] = hist_voxels->counters[i];
			}

			for(int i = 0; i < 50; i++){
				results->pointsPerLevel[i] = pointsAtLevel[i];
				results->voxelsPerLevel[i] = voxelsAtLevel[i];
				results->innerNodesAtLevel[i] = innerAtLevel[i];
				results->leafNodesAtLevel[i] = leavesAtLevel[i];
			}

			results->allocatedMemory_voxelization = allocator.offset;


			// printf("results: %llu \n", results);
		}

		// // PRINT
		// if(false)
		// if(isFirstThread())
		// { // DEBUG

		// 	printf("#points:                 ");
		// 	printNumber(numPoints, 13);
		// 	printf("\n");

		// 	printf("#voxels:                 ");
		// 	printNumber(stat_sum_inner, 13);
		// 	printf("\n");

		// 	printf("#nodes:                  "); 
		// 	printNumber(numNodes, 13);
		// 	printf("\n");

		// 	printf("    inner:               "); 
		// 	printNumber(stat_num_inner, 13);
		// 	printf("\n");

		// 	printf("    leaves:              "); 
		// 	printNumber(stat_num_leaves, 13);
		// 	printf("\n");

		// 	printf("                      min       max       avg\n");
		// 	printf("node.points:   "); 
		// 	printNumber(stat_min_leaves, 10);
		// 	printNumber(stat_max_leaves, 10);
		// 	printNumber(stat_sum_leaves / stat_num_leaves, 10);
		// 	printf("\n");

		// 	printf("node.voxels:   ");    
		// 	printNumber(stat_min_inner, 10);
		// 	printNumber(stat_max_inner, 10);
		// 	printNumber(stat_sum_inner / stat_num_inner, 10);
		// 	printf("\n");        
 
		// 	hist_points->print("HISTOGRAM - POINTS", 5);
		// 	hist_voxels->print("HISTOGRAM - VOXELS", 5); 

		// 	printf("== level stats =========================\n");
		// 	printf("level    inner    leaves        #points        #voxels      rate\n");
		// 	for(int i = 0; i < 14; i++){
		// 		int numPoints = pointsAtLevel[i];
		// 		int numVoxels = voxelsAtLevel[i];
		// 		int samples_0 = numPoints + numVoxels;
		// 		int samples_1 = pointsAtLevel[i + 1] + voxelsAtLevel[i + 1];
		// 		int numInnerNodes = innerAtLevel[i];
		// 		int numLeafNodes  = leavesAtLevel[i];

		// 		if(samples_0 > 0){
		// 			printf("%2i: ", i);

		// 			printNumber(numInnerNodes, 10);
		// 			printNumber(numLeafNodes, 10);
		// 			printNumber(numPoints, 15);
		// 			printNumber(numVoxels, 15);

		// 			// printNumber(samples_0, 14);

		// 			if(i < 14){
		// 				float percentage = float(samples_0) / float(samples_1);
		// 				int percentage_i = 100.0f * percentage;

		// 				printf("   %5i %%", percentage_i);
		// 			}

		// 			printf("\n");
		// 		}
		// 	}
		// 	printf("======================================\n");

		// 	printf("allocated memory:        "); //%i MB \n", allocator.offset / (1024 * 1024));
		// 	printNumber(allocator.offset, 13);
		// 	printf("\n");
		// }
	} 

		// void** nodes,
	// uint32_t* num_nodes,
	// PRINT("uint32_t* buffer: %llu \n", buffer);
	// PRINT("void* nodes: %llu \n", *nodes);
	// PRINT("offsetToNodeArray: %llu \n", uint64_t(*nodes) - uint64_t(buffer));

	

}

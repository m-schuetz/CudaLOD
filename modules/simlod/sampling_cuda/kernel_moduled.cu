
#include <cooperative_groups.h>
#include "kernel_data.h.cu"
#include "kernel_lib.h.cu"

namespace cg = cooperative_groups;

#define WIDGET_TYPE_SLIDER 1

#define POINT_SIZE 4

// #define SAMPLING_METHOD_NEAREST 1
// #define SAMPLING_METHOD_LINEAR 2
// #define SAMPLING_METHOD_ORIGINAL 3

// #define SAMPLING_METHOD SAMPLING_METHOD_LINEAR 
// #define SAMPLING_METHOD SAMPLING_METHOD_NEAREST 
// #define SAMPLING_METHOD SAMPLING_METHOD_ORIGINAL 

#define BATCHES_PER_BLOCK 1000
#define POINTS_PER_BATCH 1000
#define NUM_BATCHES 50'000
#define MAX_OCTREE_DEPTH 8
#define MAX_POINTS_PER_NODE 50 * 1000

#define VERBOSE false

#define VERBOSE_PRINT(...) {                     \
	if(cg::this_grid().thread_rank() == 0 && VERBOSE){ \
		printf(__VA_ARGS__);                \
	}                                       \
}

#define PRINT(...) {                     \
	if(cg::this_grid().thread_rank() == 0){ \
		printf(__VA_ARGS__);                \
	}                                       \
}


// calls function <f> <size> times
// calls are distributed over all available threads
template<typename Function>
void parallelIterator(int first, int size, Function&& f){

	uint32_t totalThreadCount = blockDim.x * gridDim.x;

	// TODO: make sure there are no duplicate calls
	int itemsPerThread = size / totalThreadCount + 1;

	for(int i = 0; i < itemsPerThread; i++){
		int block_offset  = itemsPerThread * blockIdx.x * blockDim.x;
		int thread_offset = itemsPerThread * threadIdx.x;
		int index = first + block_offset + thread_offset + i;

		if(index >= first + size){
			break;
		}

		f(index);
	}
}


int dbg;

struct Allocator{

	uint8_t* buffer = nullptr;
	int64_t offset = 0;

	Allocator(unsigned int* buffer, int64_t offset){
		this->buffer = reinterpret_cast<uint8_t*>(buffer);
		this->offset = offset;
	}

	template<class T>
	T alloc(int64_t size){

		auto ptr = reinterpret_cast<T>(buffer + offset);

		int64_t newOffset = offset + size;
		
		// make allocated buffer location 16-byte aligned to avoid 
		// potential problems with bad alignments
		int64_t remainder = (newOffset % 16ll);

		if(remainder != 0ll){
			newOffset = (newOffset - remainder) + 16ll;
		}
		

		if(offset >= 10 * 1024 * 1024){
			PRINT("offset: %8i Mb", offset / (1024 * 1024));
		}else if(offset >= 10 * 1024){
			PRINT("offset: %8i kb", offset / 1024);
		}else{
			PRINT("offset: %8i  b", offset);
		}

		if(size >= 10 * 1024 * 1024){
			PRINT(",   size: %10i Mb", size / (1024 * 1024));
		}else if(size >= 10 * 1024){
			PRINT(",   size: %10i kb", size / 1024);
		}else{
			PRINT(",   size: %10i  b", size);
		}

		if(newOffset >= 10 * 1024 * 1024){
			PRINT(",   newOffset: %10i Mb \n", newOffset / (1024 * 1024));
		}else if(newOffset >= 10 * 1024){
			PRINT(",   newOffset: %10i kb\n", newOffset / 1024);
		}else{
			PRINT(",   newOffset: %10i  b\n", newOffset);
		}

		this->offset = newOffset;

		return ptr;
	}

};

struct Buffer{
	int offset;
	int size;
	uint8_t* ptr;
};

struct GuiItem{
	uint32_t type;
	float min;
	float max;
	float value;
	char label[32];
};

struct PointCloudMetadata{
	uint32_t numPoints;
	float min_x;
	float min_y;
	float min_z;
	float max_x;
	float max_y;
	float max_z;
};

struct SampleGrid{
	uint32_t* values;
};

struct State{
	uint32_t semaphores[64];  // 64
	uint32_t numGuiItems;     //  4
	GuiItem guiItems[16];     // 64
	uint32_t wasJustCompiled;
	PointCloudMetadata metadata;
	uint32_t debug;
};

struct Octree;

struct PointPool{

	Point points[NUM_BATCHES * POINTS_PER_BATCH];
	// 0: points 0 to 999,
	// 1: points 1000 to 1999, ...
	uint32_t batchIndices[NUM_BATCHES];
	int currentBatch;

	uint32_t acquire(){
		int index = atomicAdd(&this->currentBatch, 1);

		// if(index == 10'000){
		// 	printf("index: %i \n", index);
		// }

		if((index % 10'000) == 0){
			printf("acquire batch: %i \n", index);
		}

		if(index > NUM_BATCHES - 10){
			printf("running out of batches: %i \n", index);
		}

		if(index >= NUM_BATCHES){
			printf("no more batches, from now on garbage: %i \n", index);
		}

		return batchIndices[index];
	}
	
	void release(uint32_t batchIndex){
		int index = atomicAdd(&this->currentBatch, -1) - 1;

		batchIndices[index] = batchIndex;
	}
	
};

struct PointBatch{
	int batchIndex;
	int numPoints;
	Point* points;
};

struct PointBatchWorkload{
	uint32_t numBatches;
	uint32_t current;
	PointBatch batches[10'000 * POINTS_PER_BATCH];
};

struct SparseSampleGrid{

	uint64_t* keys;
	uint32_t* values;

	vec3 min;
	vec3 max;
	vec3 size;
	int gridSize;
	float fGridSize;

	uint64_t max_capacity = 0;   // indicates allocated memory
	uint64_t capacity = 0;       // currently used capacity/memory

	constexpr static uint64_t EMPTY_VALUE = 0xffffffff'ffffffffull;

	SparseSampleGrid(){

	}

	void clear(){

		SparseSampleGrid* sampleGrid = this;

		// clear key buffer
		parallelIterator(0, capacity, [sampleGrid](int mapIndex){
			sampleGrid->keys[2 * mapIndex + 0] = SparseSampleGrid::EMPTY_VALUE;
			sampleGrid->keys[2 * mapIndex + 1] = SparseSampleGrid::EMPTY_VALUE;
		});

		// clear value buffer
		parallelIterator(0, capacity, [sampleGrid](int mapIndex){
			sampleGrid->values[4ll * mapIndex + 0ll] = 0;
			sampleGrid->values[4ll * mapIndex + 1ll] = 0;
			sampleGrid->values[4ll * mapIndex + 2ll] = 0;
			sampleGrid->values[4ll * mapIndex + 3ll] = 0;
		});
	}

	void add_128bit(Point point){

		// 128 bit add

		float fx = fGridSize * (point.x) / size.x;
		float fy = fGridSize * (point.y) / size.y;
		float fz = fGridSize * (point.z) / size.z;

		uint64_t ix = uint64_t(clamp(fx, 0.0, fGridSize - 1.0));
		uint64_t iy = uint64_t(clamp(fy, 0.0, fGridSize - 1.0));
		uint64_t iz = uint64_t(clamp(fz, 0.0, fGridSize - 1.0));

		// uint64_t voxelIndex = ix | (iy << 20ull) | (iz << 40ull);

		uint64_t voxelIndex_0 = 
		      ((ix & 0x0000FFFFull) <<  0ull)
		    | ((iy & 0x0000FFFFull) << 20ull)
		    | ((iz & 0x0000FFFFull) << 40ull);

		uint64_t voxelIndex_1 = 
		      ((ix & 0xFFFF0000ull) >> 16ull)
		    | ((iy & 0xFFFF0000ull) <<  4ull)
		    | ((iz & 0xFFFF0000ull) << 24ull);

		for(uint64_t i = 0; i < 100; i++)
		{

			uint64_t mapIndex = (voxelIndex_0 * 13ull + voxelIndex_1 * 13ull + i) % capacity;
			// uint64_t mapIndex = (voxelIndex * 13ull + i) % capacity;

			// attempt non-atomic map key retrieval first (~15% faster)
			// if equal to voxelIndex, we're good
			// otherwise, we need to do atomicCAS
			// uint64_t old = keys[mapIndex];
			// if(old != voxelIndex){
			// 	old = atomicCAS(keys + mapIndex, SparseSampleGrid::EMPTY_VALUE, voxelIndex);
			// }

			// always doing atomicCAS is slightly slower
			uint64_t old_0 = atomicCAS(keys + 2 * mapIndex + 0, SparseSampleGrid::EMPTY_VALUE, voxelIndex_0);

			if(old_0 == SparseSampleGrid::EMPTY_VALUE || old_0 == voxelIndex_0){
				// try 2nd key part

				uint64_t old_1 = atomicCAS(keys + 2 * mapIndex + 1, SparseSampleGrid::EMPTY_VALUE, voxelIndex_1);

				if(old_1 == SparseSampleGrid::EMPTY_VALUE || old_1 == voxelIndex_1){
					// map location was empty, inserted current key and init values

					uint32_t R = (point.color >>  0) & 0xFF;
					uint32_t G = (point.color >>  8) & 0xFF;
					uint32_t B = (point.color >> 16) & 0xFF;

					atomicAdd(this->values + (4 * mapIndex + 0), R);
					atomicAdd(this->values + (4 * mapIndex + 1), G);
					atomicAdd(this->values + (4 * mapIndex + 2), B);
					atomicAdd(this->values + (4 * mapIndex + 3), 1u);

					// // if(atomicAdd(&globals.state->semaphores[9], 1) == 0){
					// if(atomicCAS(&globals.state->semaphores[9], 0, mapIndex) == 0){
					// 	printf("ixyz: %i", ix);
					// 	printf(", %i", iy);
					// 	printf(", %i \n", iz);
					// 	printf("voxel indices: %i, %i \n", voxelIndex_0, voxelIndex_1);
					// }

					break;
				}else{
					// 2nd part taken by other key. do probing
				}

			}

			// if we didn't break the loop, we had a collision and will probe 
			// for another spot

		}


	}

	void add(Point point){

		float fx = fGridSize * (point.x) / size.x;
		float fy = fGridSize * (point.y) / size.y;
		float fz = fGridSize * (point.z) / size.z;

		uint64_t ix = uint64_t(clamp(fx, 0.0, fGridSize - 1.0));
		uint64_t iy = uint64_t(clamp(fy, 0.0, fGridSize - 1.0));
		uint64_t iz = uint64_t(clamp(fz, 0.0, fGridSize - 1.0));

		uint64_t voxelIndex = ix | (iy << 20ull) | (iz << 40ull);

		for(uint64_t i = 0; i < 100; i++)
		{

			uint64_t mapIndex = (voxelIndex * 13ull + 12345ull * i) % capacity;

			// attempt non-atomic map key retrieval first (~15% faster)
			// if equal to voxelIndex, we're good
			// otherwise, we need to do atomicCAS
			uint64_t old = keys[mapIndex];
			if(old != voxelIndex){
				old = atomicCAS(keys + mapIndex, SparseSampleGrid::EMPTY_VALUE, voxelIndex);
			}

			// always doing atomicCAS is slightly slower
			// uint64_t old = atomicCAS(keys + mapIndex, SparseSampleGrid::EMPTY_VALUE, voxelIndex);

			if(old == SparseSampleGrid::EMPTY_VALUE || old == voxelIndex){
				// map location was empty, inserted current key and init values

				uint32_t R = (point.color >>  0) & 0xFF;
				uint32_t G = (point.color >>  8) & 0xFF;
				uint32_t B = (point.color >> 16) & 0xFF;

				atomicAdd(this->values + (4 * mapIndex + 0), R);
				atomicAdd(this->values + (4 * mapIndex + 1), G);
				atomicAdd(this->values + (4 * mapIndex + 2), B);
				auto old = atomicAdd(this->values + (4 * mapIndex + 3), 1u);

				// if(voxelIndex == 10995179192382ull){
				// 	printf("mapIndex: %i \n", mapIndex);
				// 	printf("%i, %i, %i, %i \n", R, G, B, 1);
				// 	printf("old: %i \n", old);
				// }

				break;
			}
			// else if(old == voxelIndex){
			// 	// map location contains current key. update values

			// 	uint32_t R =(point.color >>  0) & 0xFF;
			// 	uint32_t G =(point.color >>  8) & 0xFF;
			// 	uint32_t B =(point.color >> 16) & 0xFF;

			// 	atomicAdd(this->values + (4 * mapIndex + 0), R);
			// 	atomicAdd(this->values + (4 * mapIndex + 1), G);
			// 	atomicAdd(this->values + (4 * mapIndex + 2), B);
			// 	atomicAdd(this->values + (4 * mapIndex + 3), 1u);

			// 	break;
			// }

			// if we didn't break the loop, we had a collision and will probe 
			// for another spot

		}


	}



};


struct {
	unsigned long long int* framebuffer;
	unsigned int* buffer;
	Point* input_points;
	Points* points;
	Lines* lines;
	Triangles* triangles;
	State* state;
	Octree* octree;
	PointPool* pool;
	Allocator* allocator;
	SparseSampleGrid* sparseGrid;
} globals;

struct Node{
	uint32_t numPoints;
	uint32_t padding;
	uint32_t numNewPoints;
	uint32_t padding2;
	uint32_t childrenMask;
	uint32_t childrenIndex;
	uint32_t numBatches;
	uint32_t index;
	int level;
	Box3 boundingBox;
	int32_t batchIndices[200];
	int id;
	int name[20];

	uint32_t voxelBatchIndices[200];
	uint32_t numVoxelBatches;
	uint32_t numVoxels;
	uint32_t numNewVoxels;

	Node(){
		numPoints = 0;
		numNewPoints = 0;
		childrenMask = 0;
		childrenIndex = 0;
		numBatches = 0;
		index = 0;
		level = 0;
		id = 0;
		numVoxels = 0;
		numVoxelBatches = 0;
		numNewVoxels = 0;
		memset(name, 0, 4 * 20);
	}

	int getName(){
		int name = 0;

		if(level > 0) name = name +         1 * this->name[1];
		if(level > 1) name = name +        10 * this->name[2];
		if(level > 2) name = name +       100 * this->name[3];
		if(level > 3) name = name +     1'000 * this->name[4];
		if(level > 4) name = name +    10'000 * this->name[5];
		if(level > 5) name = name +   100'000 * this->name[6];
		if(level > 7) name = name + 1'000'000 * this->name[7];
		
		return name;
	}

	bool isInside(Point point){
		
		bool insideX = (boundingBox.min.x <= point.x) && (point.x <= boundingBox.max.x);
		bool insideY = (boundingBox.min.y <= point.y) && (point.y <= boundingBox.max.y);
		bool insideZ = (boundingBox.min.z <= point.z) && (point.z <= boundingBox.max.z);
		
		return insideX && insideY && insideZ;
	}

	bool is(const char* name){

		for(int i = 0; i < 20; i++){


			int a_at = name[i];
			int b_at = this->name[i];

			if(a_at == 0) return true;

			a_at = a_at == 'r' ? a_at : a_at - '0';

			// printf("is %i, %i, %i \n", i, a_at, b_at);
			if(a_at != b_at) return false;

		}

		return true;
	}

	template<typename F>
	void traverse(F&& lambda, Octree* octree){

		if(!isFirstThread()) return;

		lambda(this);

		for(int i = 0; i < 8; i++){
			bool childExists = (childrenMask & (1 << i)) != 0;

			if(childExists){
				Node* child = &octree->nodes[childrenIndex + i];

				child->traverse(lambda, octree);
			}
		}

	}

	// print the octree hierarchy to the console
	void printRecursive(Node* nodes, int level, char string[20]){

		if(level > 10){
			return;
		}

		if(childrenMask == 0 && numPoints == 0 && numNewPoints == 0){
			return;
		}

		if(level == 0){
			string[level] = 'r';
		}else{
			string[level] = '0' + this->index;
		}
		string[level + 1] = ':';
		string[level + 2] = ' ';

		for(int i = level + 2; i < 19; i++){
			string[i] = ' ';
		}

		string[19] = 0;

		printf("    ");
		printf(string);
		printf(", id: %3i, #points: %6i, #newPoints: %6i, #batches: %3i \n", 
			id, numPoints, numNewPoints, numBatches);
		// printf("offsetof %i \n", offsetof(this->numNewPoints));

		for(int i = 0; i < 8; i++){
			bool childExists = (childrenMask & (1 << i)) != 0;

			if(childExists){
				Node* child = &nodes[childrenIndex + i];

				child->printRecursive(nodes, level + 1, string);
			}
		}
	}

};

Box3 computeChildAABB(Box3 box, int childIndex){

	Box3 childBox;
	vec3 min = box.min;
	vec3 max = box.max;
	vec3 ctr = (box.min + box.max) / 2.0;

	int cx = (childIndex >> 0) & 1;
	int cy = (childIndex >> 1) & 1;
	int cz = (childIndex >> 2) & 1;

	childBox.min.x = (cx == 0) ? min.x : ctr.x;
	childBox.min.y = (cy == 0) ? min.y : ctr.y;
	childBox.min.z = (cz == 0) ? min.z : ctr.z;

	childBox.max.x = (cx == 0) ? ctr.x : max.x;
	childBox.max.y = (cy == 0) ? ctr.y : max.y;
	childBox.max.z = (cz == 0) ? ctr.z : max.z;

	return childBox;
}

struct Octree{
	int numNodes;
	int rootIndex;
	vec3 min;
	vec3 max;
	vec3 size;
	uint32_t padding;
	Node nodes[20 * 1000];
	int gridSize;
	float fGridSize;

	Box3 boundingBoxOf(Node* node){
		Box3 box;

		return box;
	}

	Node* find(const char* name){

		int depth = strlen(name) - 1;
		Node* root = &nodes[0];

		if(depth == 0){
			return root;
		}

		Node* node = root;

		for(int level = 1; level <= depth; level++){

			int childIndex = name[level] - '0';

			if((node->childrenMask >> childIndex) == 0){
				return nullptr;
			}
			
			node = &nodes[node->childrenIndex + childIndex];
		}

		return node;
	}

	void printRecursive(const char* label = " "){

		if(isFirstThread()){ 
			printf("OCTREE[");
			printf(label);
			printf("]: \n");
		
			char string[20];
			Node* root = &nodes[0];
			root->printRecursive(&nodes[0], 0, string);
		}

	}

	Node* findLeaf(Point point, int dbg_index){
		int factor = pow(2.0f, float(MAX_OCTREE_DEPTH));

		int X = clamp(factor * ((point.x - min.x) / size.x), 0, factor - 1);
		int Y = clamp(factor * ((point.y - min.y) / size.y), 0, factor - 1);
		int Z = clamp(factor * ((point.z - min.z) / size.z), 0, factor - 1);

		// find leaf node for point
		Node* node = &nodes[rootIndex];
		uint32_t nodeIndex = rootIndex;
		for(int level = 0; level < MAX_OCTREE_DEPTH; level++){

			int shift = (MAX_OCTREE_DEPTH - level - 1);
			int cx = (X >> shift) & 1;
			int cy = (Y >> shift) & 1;
			int cz = (Z >> shift) & 1;
			int childIndex = cx | (cy << 1) | (cz << 2);

			bool hasChild = (node->childrenMask & (1 << childIndex)) != 0;

			if(hasChild){
				nodeIndex = node->childrenIndex + childIndex;
				node = &this->nodes[nodeIndex];
			}else{
				break;
			}
		}

		return node;
	}

	Node* find(Point point, int targetLevel){
		int factor = pow(2.0f, float(MAX_OCTREE_DEPTH));

		int X = clamp(factor * ((point.x - min.x) / size.x), 0, factor - 1);
		int Y = clamp(factor * ((point.y - min.y) / size.y), 0, factor - 1);
		int Z = clamp(factor * ((point.z - min.z) / size.z), 0, factor - 1);

		// find node at targetLevel for point
		Node* node = &nodes[rootIndex];
		uint32_t nodeIndex = rootIndex;
		for(int level = 0; level < targetLevel; level++){

			int shift = (MAX_OCTREE_DEPTH - level - 1);
			int cx = (X >> shift) & 1;
			int cy = (Y >> shift) & 1;
			int cz = (Z >> shift) & 1;
			int childIndex = cx | (cy << 1) | (cz << 2);

			bool hasChild = (node->childrenMask & (1 << childIndex)) != 0;

			if(hasChild){
				nodeIndex = node->childrenIndex + childIndex;
				node = &this->nodes[nodeIndex];
			}else{
				break;
			}
		}

		if(node->level != targetLevel){
			return nullptr;
		}else{
			return node;
		}

	}

	void insert_count(Point point, int dbg_pointIndex){

		// uint32_t nodeIndex = this->findLeaf(point, dbg_pointIndex);
		// Node* node = &globals.octree->nodes[nodeIndex];

		Node* node = this->findLeaf(point, dbg_pointIndex);

		atomicAdd(&node->numNewPoints, 1);
	}

	void insert(PointPool* pool, Point point, int dbg_pointIndex){

		// uint32_t nodeIndex = this->findLeaf(point, dbg_pointIndex);
		// Node* node = &globals.octree->nodes[nodeIndex];

		Node* node = this->findLeaf(point, dbg_pointIndex);

		uint32_t pointIndex = atomicAdd(&node->numPoints, 1);
		int* ptr_batchIndex = &node->batchIndices[pointIndex / POINTS_PER_BATCH];

		// write point to batch
		int batchIndex = *ptr_batchIndex;
		uint32_t point_in_batch_index = pointIndex % POINTS_PER_BATCH;
		pool->points[POINTS_PER_BATCH * batchIndex + point_in_batch_index] = point;
	}

	template<typename F>
	void traverse(F&& lambda){

		if(!isFirstThread()) return;

		Node* root = &nodes[0];

		root->traverse(lambda, this);

	}
};




void COUNT(Octree* octree, Point* input_points, PointBatchWorkload* work){

	auto timer_count = Timer::start("     COUNT");
	
	__shared__ int shared_batchIndex[1];

	auto grid  = cg::this_grid();
	auto block = cg::this_thread_block();

	block.sync();

	int loop_max = 50;
	for(int loop_i = 0; loop_i < loop_max; loop_i++){
		
		block.sync();
		if(block.thread_rank() == 0){
			shared_batchIndex[0] = atomicAdd(&work->current, 1);
		}
		block.sync();

		int batchIndex = shared_batchIndex[0];


		if(batchIndex < work->numBatches){

			auto batch = work->batches[batchIndex];
			int pointsPerThread = (batch.numPoints / blockDim.x) + 1;

			// Node* leaf = nullptr;
			// int X, Y, Z = -1;
			// for(int i = 0; i < pointsPerThread; i++){
			// 	int pointIndex = threadIdx.x + i * blockDim.x;

			// 	if(pointIndex >= batch.numPoints){
			// 		continue; // or better/safe to break?
			// 	}

			// 	Point point = batch.points[pointIndex];

			// 	int factor = pow(2.0f, float(MAX_OCTREE_DEPTH));
			// 	int nX = clamp(factor * ((point.x - octree->min.x) / octree->size.x), 0, factor - 1);
			// 	int nY = clamp(factor * ((point.y - octree->min.y) / octree->size.y), 0, factor - 1);
			// 	int nZ = clamp(factor * ((point.z - octree->min.z) / octree->size.z), 0, factor - 1);

			// 	if(nX != X || nY != Y || nZ != Z || leaf == nullptr){
			// 		leaf = octree->findLeaf(point, pointIndex);
			// 	}

			// 	atomicAdd(&leaf->numNewPoints, 1);

			// 	X = nX;
			// 	Y = nY;
			// 	Z = nZ;

			// 	// octree->insert_count(point, pointIndex);

				
			// 	// atomicAdd(&node->numNewPoints, 1);
			// }

			for(int i = 0; i < pointsPerThread; i++){
				int pointIndex = threadIdx.x + i * blockDim.x;

				if(pointIndex >= batch.numPoints){
					continue; // or better/safe to break?
				}

				Point point = batch.points[pointIndex];

				octree->insert_count(point, pointIndex);
			}
		}
	}

	Timer::stop(timer_count);

	grid.sync();
	work->current = 0;
}

void SPLIT(Octree* octree, Point* input_points, PointBatchWorkload* work){

	auto timer = Timer::start("     SPLIT");

	uint32_t totalThreadCount = blockDim.x * gridDim.x;

	parallelIterator(0, octree->numNodes, [octree, work](int nodeIndex){
		Node* node = &octree->nodes[nodeIndex];

		bool exceedsThreshold = node->numPoints + node->numNewPoints > MAX_POINTS_PER_NODE;
		if(exceedsThreshold){
			// split
			node->numNewPoints = 0;
			node->childrenIndex = atomicAdd(&octree->numNodes, 8);
			node->childrenMask = 0xFF;
			
			// if(false)
			{
				// move points/batches from this node to workload
				int batchOffset = atomicAdd(&work->numBatches, node->numBatches);

				for(int j = 0; j < node->numBatches; j++){
					PointBatch batch;
					batch.batchIndex = node->batchIndices[j];
					batch.numPoints = clamp(node->numPoints - j * POINTS_PER_BATCH, 0, POINTS_PER_BATCH);
					batch.points = globals.pool->points + batch.batchIndex * POINTS_PER_BATCH;

					work->batches[batchOffset + j] = batch;
				}

				node->numPoints = 0;
				node->numBatches = 0;
			}


			for(int childIndex = 0; childIndex < 8; childIndex++){
				Node child;
				child.index = childIndex;
				child.level = node->level + 1;
				child.id = node->childrenIndex + childIndex;
				child.boundingBox = computeChildAABB(node->boundingBox, childIndex);
				memcpy(child.name, node->name, 4 * child.level);
				child.name[child.level] = child.index;
				
				octree->nodes[node->childrenIndex + childIndex] = child;
			}
		}

		node->numNewPoints = 0;
	});

	Timer::stop(timer);
}
 

struct Voxels{
	int numVoxels;
};

Voxels sampleVoxels(PointBatchWorkload* work, int level){

	auto grid = cg::this_grid();

	int gridSize = 128 * pow(2.0f, float(level));
	auto metadata   = globals.state->metadata;
	auto sampleGrid = globals.sparseGrid;

	sampleGrid->gridSize  = gridSize;
	sampleGrid->fGridSize = gridSize;
	sampleGrid->capacity  = 1'000'016;

	auto timer_clear = Timer::start("    sampleVoxels - clear");
	sampleGrid->clear();
	Timer::stop(timer_clear);

	grid.sync();

	auto timer_addtogrid = Timer::start("    sampleVoxels - add");
	parallelIterator(0, POINTS_PER_BATCH * work->numBatches, [sampleGrid, work](int pointIndex){

		int batchIndex = pointIndex / POINTS_PER_BATCH;
		int localPointIndex = pointIndex % POINTS_PER_BATCH;
		PointBatch& batch = work->batches[batchIndex];

		if(localPointIndex >= batch.numPoints) return;

		Point& point = batch.points[localPointIndex];

		sampleGrid->add(point);
	});
	Timer::stop(timer_addtogrid);

	grid.sync();

	if(grid.thread_rank() == 0){
		atomicAdd(&globals.state->semaphores[13], 1);
	}

	grid.sync();

	int iteration = globals.state->semaphores[13];

	// PRINT("iteration: %i \n", iteration);

	// extract voxels from sample grid
	// Timer::timestamp("start extracting voxels");

	// count new voxels in tareged nodes
	auto timer_countnew = Timer::start("    sampleVoxels - count");
	parallelIterator(0, sampleGrid->capacity, [sampleGrid, iteration, level](int mapIndex){

		uint64_t key = sampleGrid->keys[mapIndex];

		if(key == SparseSampleGrid::EMPTY_VALUE) return;

		uint32_t ix = (key >>  0ull) & 0b1111'11111111'11111111ull;
		uint32_t iy = (key >> 20ull) & 0b1111'11111111'11111111ull;
		uint32_t iz = (key >> 40ull) & 0b1111'11111111'11111111ull;

		float x = (float(ix) / sampleGrid->fGridSize) * sampleGrid->size.x; // + sampleGrid.min.x;
		float y = (float(iy) / sampleGrid->fGridSize) * sampleGrid->size.y; // + sampleGrid.min.y;
		float z = (float(iz) / sampleGrid->fGridSize) * sampleGrid->size.z; // + sampleGrid.min.z;

		uint32_t R = sampleGrid->values[4 * mapIndex + 0];
		uint32_t G = sampleGrid->values[4 * mapIndex + 1];
		uint32_t B = sampleGrid->values[4 * mapIndex + 2];
		uint32_t C = sampleGrid->values[4 * mapIndex + 3];

		int r = (R / C) & 0xFF;
		int g = (G / C) & 0xFF;
		int b = (B / C) & 0xFF;

		uint32_t color = r | (g << 8) | (b << 16);

		color = iteration * 1234567;

		vec3 pos = {x, y, z};

		// drawPoint(globals.points, pos, color);

		Point voxel;
		voxel.x = pos.x;
		voxel.y = pos.y;
		voxel.z = pos.z;
		voxel.color = 0x000000ff;
		Node* node = globals.octree->find(voxel, level);

		// int voxelIndex = atomicAdd(&node->numVoxels, 1);

		// drawPoint(globals.points, pos, node->id * 12345678);

		if(node){
			atomicAdd(&node->numNewVoxels, 1);
		}
	});

	Timer::stop(timer_countnew);

	grid.sync();

	// allocate enough voxel batches for targeted nodes
	auto timer_allocate = Timer::start("    sampleVoxels - alloc");
	parallelIterator(0, globals.octree->numNodes, [](int index){

		Node* node = &globals.octree->nodes[index];
		int numRequiredBatches = ceil(float(node->numVoxels + node->numNewVoxels) / float(POINTS_PER_BATCH));

		// allocate new batches
		for(int j = node->numVoxelBatches; j < numRequiredBatches; j++){
			uint32_t batchIndex = globals.pool->acquire();
			node->voxelBatchIndices[j] = batchIndex;
		}


		// if(node->numNewVoxels > 0){
		// 	printf("id: %i, #numVoxelBatches: %i \n", node->id, node->numVoxelBatches);
		// }

		// node->numVoxels = node->numVoxels + node->numNewVoxels;
		
		node->numVoxelBatches = numRequiredBatches;
		node->numNewVoxels = 0;
	});

	Timer::stop(timer_allocate);

	grid.sync();

	// add voxels to targeted nodes
	auto timer_addvoxels = Timer::start("    sampleVoxels - insert");
	parallelIterator(0, sampleGrid->capacity, [sampleGrid, iteration, level](int mapIndex){

		uint64_t key = sampleGrid->keys[mapIndex];

		if(key == SparseSampleGrid::EMPTY_VALUE) return;

		uint32_t ix = (key >>  0ull) & 0b1111'11111111'11111111ull;
		uint32_t iy = (key >> 20ull) & 0b1111'11111111'11111111ull;
		uint32_t iz = (key >> 40ull) & 0b1111'11111111'11111111ull;

		float x = (float(ix) / sampleGrid->fGridSize) * sampleGrid->size.x; // + sampleGrid.min.x;
		float y = (float(iy) / sampleGrid->fGridSize) * sampleGrid->size.y; // + sampleGrid.min.y;
		float z = (float(iz) / sampleGrid->fGridSize) * sampleGrid->size.z; // + sampleGrid.min.z;

		uint32_t R = sampleGrid->values[4 * mapIndex + 0];
		uint32_t G = sampleGrid->values[4 * mapIndex + 1];
		uint32_t B = sampleGrid->values[4 * mapIndex + 2];
		uint32_t C = sampleGrid->values[4 * mapIndex + 3];

		int r = (R / C) & 0xFF;
		int g = (G / C) & 0xFF;
		int b = (B / C) & 0xFF;

		uint32_t color = r | (g << 8) | (b << 16);

		color = iteration * 1234567;

		vec3 pos = {x, y, z};

		Point voxel;
		voxel.x = pos.x;
		voxel.y = pos.y;
		voxel.z = pos.z;
		voxel.color = 0x000000ff;
		Node* node = globals.octree->find(voxel, level);

		if(node){
			int voxelIndex = atomicAdd(&node->numVoxels, 1);
			int batchIndexIndex = voxelIndex / POINTS_PER_BATCH;
			int localVoxelIndex = voxelIndex % POINTS_PER_BATCH;

			int batchIndex = node->voxelBatchIndices[batchIndexIndex];
			globals.pool->points[POINTS_PER_BATCH * batchIndex + localVoxelIndex] = voxel;
		}
	});

	Timer::stop(timer_addvoxels);

}

void insert(PointBatchWorkload* work){

	// #if defined(VERBOSE) 
	VERBOSE_PRINT("insert[%i] \n", dbg);
	
	uint32_t totalThreadCount = blockDim.x * gridDim.x;

	Octree* octree        = globals.octree;
	Point* input_points   = globals.input_points;
	// State* state          = globals.state;
	// Triangles* triangles  = globals.triangles;
	PointPool* pool       = globals.pool;
	// Points* points        = globals.points;

	auto grid = cg::this_grid();
	grid.sync();
	
	// count points in leaf nodes
	// split leaf nodes with too many points
	// repeat a couple of times. 
	// real insertion of points starts later

	// Timer::timestamp("start counting");
	auto timer_count_split = Timer::start("  insert - count&split");
	COUNT(octree, input_points, work); grid.sync(); // timer->timestamp("COUNT");
	SPLIT(octree, input_points, work); grid.sync(); // timer->timestamp("SPLIT");
	COUNT(octree, input_points, work); grid.sync(); // timer->timestamp("COUNT");
	SPLIT(octree, input_points, work); grid.sync(); // timer->timestamp("SPLIT");
	COUNT(octree, input_points, work); grid.sync(); // timer->timestamp("COUNT");
	SPLIT(octree, input_points, work); grid.sync(); // timer->timestamp("SPLIT");
	COUNT(octree, input_points, work); grid.sync(); // timer->timestamp("COUNT");
	SPLIT(octree, input_points, work); grid.sync(); // timer->timestamp("SPLIT");
	COUNT(octree, input_points, work); grid.sync(); // timer->timestamp("COUNT");
	Timer::stop(timer_count_split);
	// Timer::timestamp("end counting");

	if(dbg == 123)
	if(isFirstThread()) printf("work->numBatches: %i \n", work->numBatches);

	// ALLOCATE BATCHES
	auto timer_allocate = Timer::start("  insert - allocate");
	parallelIterator(0, octree->numNodes, [octree, pool](int index){

		Node* node = &octree->nodes[index];
		int numRequiredBatches = ceil(float(node->numPoints + node->numNewPoints) / float(POINTS_PER_BATCH));

		// allocate new batches
		for(int j = node->numBatches; j < numRequiredBatches; j++){
			uint32_t batchIndex = pool->acquire();
			node->batchIndices[j] = batchIndex;
		}

		node->numBatches = numRequiredBatches;
		node->numNewPoints = 0;
	});

	Timer::stop(timer_allocate);

	grid.sync();

	// Timer::timestamp("end allocating");

	auto timer_actual_insert = Timer::start("  insert - insert");
	{ // INSERT POINTS INTO NEWLY ALLOCATED BATCHES
		__shared__ int shared_batchIndex[1];

		auto block = cg::this_thread_block();

		block.sync();

		int loop_max = 50;
		for(int loop_i = 0; loop_i < loop_max; loop_i++){
			block.sync();

			if(threadIdx.x == 0){
				shared_batchIndex[0] = atomicAdd(&work->current, 1);
			}

			block.sync();

			int batchIndex = shared_batchIndex[0];

			if(batchIndex < work->numBatches){

				auto batch = work->batches[batchIndex];

				int pointsPerThread = (batch.numPoints / blockDim.x) + 1;

				for(int i = 0; i < pointsPerThread; i++){
					int pointIndex = threadIdx.x + i * blockDim.x;

					if(pointIndex >= batch.numPoints){
						continue; // or better/safe to break?
					}

					Point point = batch.points[pointIndex];

					octree->insert(pool, point, pointIndex);
				}

				// release batches/points from previous-leaf-now-inner nodes
				if(batchIndex >= BATCHES_PER_BLOCK && block.thread_rank() == 0){
					globals.pool->release(batch.batchIndex);
				}
			}

			
		}

		auto grid = cg::this_grid();
		grid.sync();
		work->current = 0;
	}

	Timer::stop(timer_actual_insert);

	// Timer::timestamp("end inserting");

	grid.sync();


	auto timer_sample = Timer::start("  insert - sample voxels");
	sampleVoxels(work, 0);
	grid.sync();
	sampleVoxels(work, 1);
	grid.sync();
	sampleVoxels(work, 2); 

	Timer::stop(timer_sample);

	// {
	// 	int level = 1;
	// 	int gridSize = 128 * pow(2.0f, float(level));
	// 	Voxels voxels = sampleVoxels(work, gridSize);
	// }

	// {
	// 	int level = 2;
	// 	int gridSize = 128 * pow(2.0f, float(level));
	// 	Voxels voxels = sampleVoxels(work, gridSize);
	// }

 
	// TODO 
	// - sparse-sample points
	//   - grid-size = 128 * pow(2.0, leaf->level)
	// - transfer sampled voxels to octree nodes
	//   - hash map of (affected) nodes
	//   - key ?
	//     - node-id. r123 => 0b'...'001'010'011
	//     - (node->level << 54) | (cell_x << 0) | (cell_y << 16) | (cell_z << 32)
	//   - or use octree->find method to find the correct node for each voxel


}

void main_topDownSampling(State* state, PointBatchWorkload* work, Octree* octree){

	uint32_t totalThreadCount = blockDim.x * gridDim.x;
	auto grid = cg::this_grid();

	auto input_points = globals.input_points;
	auto points       = globals.points;
	auto lines        = globals.lines;
	auto triangles    = globals.triangles;
	auto pool         = globals.pool;

	grid.sync();
	state->metadata.numPoints = 10'000'000;
	grid.sync();

	auto timer_main_insert = Timer::start("main - insert");
	// if(false)
	{ // INSERT POINTS

		int totalPoints = state->metadata.numPoints;
		int numBlocks = 1 + totalPoints / 1'000'000;
		// numBlocks = 21;

		int processedPoints = 0;
		for(int pointBlock = 0; pointBlock < numBlocks; pointBlock++){

			// create workload
			if(grid.thread_rank() == 0){
				work->numBatches = 0;
				work->current = 0;

				int block_first_batch = BATCHES_PER_BLOCK * pointBlock;
				int block_end = BATCHES_PER_BLOCK * (pointBlock + 1);
				for(int i = block_first_batch; i < block_end; i++){

					if(processedPoints >= totalPoints) break;

					int numPointsInBatch = clamp(totalPoints - processedPoints, 0, POINTS_PER_BATCH);

					PointBatch batch;
					batch.batchIndex = -1;
					batch.numPoints = numPointsInBatch;
					batch.points = input_points + (i * POINTS_PER_BATCH);

					work->batches[work->numBatches] = batch;
					work->numBatches++;

					processedPoints += batch.numPoints;

				}
			}

			grid.sync();
			dbg = pointBlock;
			insert(work);
			grid.sync();

		}
	}
	Timer::stop(timer_main_insert);

	// return;

	Timer::timestamp("inserted");


	auto timer_main_visualize = Timer::start("main - visualize");
	// VISUALIZE POINTS
	// if(false)
	{
		grid.sync();

		// ALL NODES
		parallelIterator(0, octree->numNodes, [octree, pool, points, lines](int nodeIndex){
			Node* node = &octree->nodes[nodeIndex];

			if(node->numPoints > 0)
			{ // BOX
				vec3 pos  = node->boundingBox.center();
				vec3 size = node->boundingBox.size();
				drawBoundingBox(lines, pos, size, 0xff00ffff);
			}

			// POINTS
			// for(int j = 0; j < node->numPoints; j++){
			// 	uint32_t batchIndex = node->batchIndices[j / POINTS_PER_BATCH];
			// 	uint32_t localIndex = j % POINTS_PER_BATCH;
			// 	Point point = pool->points[POINTS_PER_BATCH * batchIndex + localIndex];

			// 	uint32_t color = node->getName() * 12345678;

			// 	// if((j % 10) == 0){
			// 	// 	drawPoint(points, vec3(point.x, point.y, point.z), color);
			// 	// }
			// }

			// VOXELS
			for(int j = 0; j < node->numVoxels; j++){
				uint32_t batchIndex = node->voxelBatchIndices[j / POINTS_PER_BATCH];
				uint32_t localIndex = j % POINTS_PER_BATCH;
				Point point = pool->points[POINTS_PER_BATCH * batchIndex + localIndex];

				uint32_t color = node->getName() * 12345678;

				float zRange = globals.state->metadata.max_z - globals.state->metadata.min_z;
				float zOffset = float(node->level) * (zRange / 2.0);

				int size = 20 / int(pow(2.0f, float(node->level)));

				color = (color & 0x00ffffff) | (size << 24);

				drawPoint(points, vec3(point.x, point.y, point.z + zOffset), color);

				// if((j % 10) == 0){
				// 	drawPoint(points, vec3(point.x, point.y, point.z), color);
				// }
			}
		});


		// if(isFirstThread())
		// { // SPECIFIC NODES

		// 	// DEBUG - show contents of specific node
		// 	Node* node = octree->find("r10");

		// 	if(node){
		// 		{ // BOX
		// 			vec3 pos  = node->boundingBox.center();
		// 			vec3 size = node->boundingBox.size();
		// 			drawBoundingBox(lines, pos, size * 1.01, 0xff00ffff);
		// 		}

		// 		// // POINTS
		// 		// for(int j = 0; j < node->numPoints; j++){
		// 		// 	uint32_t batchIndex = node->batchIndices[j / POINTS_PER_BATCH];
		// 		// 	uint32_t localIndex = j % POINTS_PER_BATCH;
		// 		// 	Point point = pool->points[POINTS_PER_BATCH * batchIndex + localIndex];

		// 		// 	drawPoint(points, vec3(point.x, point.y, point.z), 12345679 * node->id);
		// 		// }

		// 		// VOXELS
		// 		for(int j = 0; j < node->numVoxels; j++){
		// 			uint32_t batchIndex = node->voxelBatchIndices[j / POINTS_PER_BATCH];
		// 			uint32_t localIndex = j % POINTS_PER_BATCH;
		// 			Point point = pool->points[POINTS_PER_BATCH * batchIndex + localIndex];

		// 			drawPoint(points, vec3(point.x, point.y, point.z), 12345679 * node->id);
		// 		}
		// 	}
		// }

		// if(false)
		// if(isFirstThread())
		// { // SPECIFIC NODES

		// 	// DEBUG - show contents of specific node
		// 	Node* node = octree->find("r67");

		// 	if(node){

		// 		printf("node->numPoints: %i \n", node->numPoints);

		// 		{ // BOX
		// 			vec3 pos  = node->boundingBox.center();
		// 			vec3 size = node->boundingBox.size();
		// 			drawBoundingBox(lines, pos, size * 1.01, 0xff00ffff);
		// 		}

		// 		// POINTS
		// 		for(int j = 0; j < node->numPoints; j++){
		// 			uint32_t batchIndex = node->batchIndices[j / POINTS_PER_BATCH];
		// 			uint32_t localIndex = j % POINTS_PER_BATCH;
		// 			Point point = pool->points[POINTS_PER_BATCH * batchIndex + localIndex];

		// 			// if((j % 100) == 0){
		// 			drawPoint(points, vec3(point.x, point.y, point.z), 12345679 * node->id);
		// 			// }
		// 		}
		// 	}
		// }

		// octree->printRecursive();

		grid.sync();

		// DISPLAY OCTREE NODES
		// octree->traverse([octree, lines](Node* node){
		// 	vec3 pos  = node->boundingBox.center();
		// 	vec3 size = node->boundingBox.size();

		// 	bool isEmpty = (node->childrenMask == 0 && node->numPoints == 0);

		// 	if(!isEmpty){
		// 		drawBoundingBox(lines, pos, size, 0xff0000ff);
		// 	}
		// });
	} 

	Timer::stop(timer_main_visualize);
}

void main_sparseGridSampling(){

	auto grid = cg::this_grid();

	int64_t gridSize = 128;
	// int64_t gridSize = 1024 * 1024;
	// int64_t gridSize = 2 * 1024 * 1024;

	auto metadata   = globals.state->metadata;
	auto sampleGrid = globals.sparseGrid;

	sampleGrid->gridSize  = gridSize;
	sampleGrid->fGridSize = gridSize;
	sampleGrid->capacity  = 2'000'017;
	
	// return;
	sampleGrid->clear();

	grid.sync();

	Timer::timestamp("start voxel sampling");

	uint64_t numPoints = 10'000'000;
	// uint64_t numPoints = metadata.numPoints;


	// add points into sample grid
	parallelIterator(0, numPoints, [sampleGrid](int pointIndex){
		Point point = globals.input_points[pointIndex];
		sampleGrid->add(point);
		// sampleGrid.add_128bit(point);
	});

	grid.sync();

	// // extract 128bit voxels from sample grid
	// Timer::timestamp("start extracting voxels");
	// parallelIterator(0, sampleGrid.capacity, [&sampleGrid](int mapIndex){

	// 	uint64_t voxelIndex_0 = sampleGrid.keys[2 * mapIndex + 0];
	// 	uint64_t voxelIndex_1 = sampleGrid.keys[2 * mapIndex + 1];

	// 	if(voxelIndex_0 == SparseSampleGrid::EMPTY_VALUE) return;
	// 	if(voxelIndex_1 == SparseSampleGrid::EMPTY_VALUE) return;

	// 	uint64_t x_0 = ((voxelIndex_0 >>  0ull) & 0xffffull);
	// 	uint64_t x_1 = ((voxelIndex_1 >>  0ull) & 0xffffull);
	// 	uint64_t y_0 = ((voxelIndex_0 >> 20ull) & 0xffffull);
	// 	uint64_t y_1 = ((voxelIndex_1 >> 20ull) & 0xffffull);
	// 	uint64_t z_0 = ((voxelIndex_0 >> 40ull) & 0xffffull);
	// 	uint64_t z_1 = ((voxelIndex_1 >> 40ull) & 0xffffull);

	// 	uint32_t ix = x_0 | (x_1 << 16);
	// 	uint32_t iy = y_0 | (y_1 << 16);
	// 	uint32_t iz = z_0 | (z_1 << 16);

	// 	// // if(atomicAdd(&globals.state->semaphores[10], 1) == 0)
	// 	// if(globals.state->semaphores[9] == mapIndex)
	// 	// {
	// 	// 	printf("voxel indices: %i, %i \n", voxelIndex_0, voxelIndex_1);
	// 	// 	// printf("x0: %i, x1: %i \n", x_0, x_1);
	// 	// 	printf("ixyz: %i", ix);
	// 	// 	printf(", %i", iy);
	// 	// 	printf(", %i \n", iz);
	// 	// }

	// 	float x = (float(ix) / sampleGrid.fGridSize) * sampleGrid.size.x; // + sampleGrid.min.x;
	// 	float y = (float(iy) / sampleGrid.fGridSize) * sampleGrid.size.y; // + sampleGrid.min.y;
	// 	float z = (float(iz) / sampleGrid.fGridSize) * sampleGrid.size.z; // + sampleGrid.min.z;

	// 	uint32_t R = sampleGrid.values[4 * mapIndex + 0];
	// 	uint32_t G = sampleGrid.values[4 * mapIndex + 1];
	// 	uint32_t B = sampleGrid.values[4 * mapIndex + 2];
	// 	uint32_t C = sampleGrid.values[4 * mapIndex + 3];

	// 	int r = (R / C) & 0xFF;
	// 	int g = (G / C) & 0xFF;
	// 	int b = (B / C) & 0xFF;

	// 	uint32_t color = r | (g << 8) | (b << 16);

	// 	vec3 pos = {x, y, z};
	// 	drawPoint(globals.points, pos, color);
	// });

	// extract voxels from sample grid
	Timer::timestamp("start extracting voxels");
	parallelIterator(0, sampleGrid->capacity, [sampleGrid](int mapIndex){

		uint64_t key = sampleGrid->keys[mapIndex];

		if(key == SparseSampleGrid::EMPTY_VALUE) return;

		uint32_t ix = (key >>  0ull) & 0b1111'11111111'11111111ull;
		uint32_t iy = (key >> 20ull) & 0b1111'11111111'11111111ull;
		uint32_t iz = (key >> 40ull) & 0b1111'11111111'11111111ull;

		float x = (float(ix) / sampleGrid->fGridSize) * sampleGrid->size.x; // + sampleGrid.min.x;
		float y = (float(iy) / sampleGrid->fGridSize) * sampleGrid->size.y; // + sampleGrid.min.y;
		float z = (float(iz) / sampleGrid->fGridSize) * sampleGrid->size.z; // + sampleGrid.min.z;

		uint32_t R = sampleGrid->values[4 * mapIndex + 0];
		uint32_t G = sampleGrid->values[4 * mapIndex + 1];
		uint32_t B = sampleGrid->values[4 * mapIndex + 2];
		uint32_t C = sampleGrid->values[4 * mapIndex + 3];

		int r = (R / C) & 0xFF;
		int g = (G / C) & 0xFF;
		int b = (B / C) & 0xFF;

		uint32_t color = r | (g << 8) | (b << 16);

		vec3 pos = {x, y, z};

		// if((color & 0x000000FF) > 250 && (color >> 16) < 100)
		// if(key == 10995179192382ull)
		// {
		// 	drawPoint(globals.points, pos, color);
		// 	// printf("key: %llu \n", key);
		// 	printf("%i, %i, %i, %i \n", R, G, B, C);
		// }

		// atomicAdd(&globals.state->semaphores[12], 1);
		drawPoint(globals.points, pos, color);
	});

	grid.sync();

	// PRINT("#voxels: %i \n", globals.points->count);
	// PRINT("globals.state->semaphores[12]: %i \n", globals.state->semaphores[12]);
	// PRINT("#points: %i \n", globals.points->count);

	// return;

	Timer::timestamp("done extracting voxels");


}

extern "C" __global__
void kernel(const CudaProgramSettings data,
			unsigned long long int* framebuffer,
			unsigned int* buffer,
			Point* input_points,
			Points* points,
			Lines* lines,
			Triangles* triangles){

	if(isFirstThread()){
		printf("=============================================\n");
		printf("=== START OCTREE GENERATION\n");
		printf("=============================================\n");
	}

	uint32_t totalThreadCount = blockDim.x * gridDim.x;
	auto grid = cg::this_grid();


	// int64_t t_start_nano = 0;
	// asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(Timer::instance->t_start_nano));
	// PRINT("start: %f \n", double(start) / 1'000'000'000.0);


	Timer::timestamp("start");
	
	if(grid.thread_rank() == 0){
		Timer::instance->t_start = clock64();
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(Timer::instance->t_start_nano));
	}

	Allocator allocator(buffer, 0);

	int maxGridCapacity = 10'000'000;

	// ALLOCATE MEMORY
	auto state           = allocator.alloc< State*              >(sizeof(State));
	auto octree          = allocator.alloc< Octree*             >(sizeof(Octree));
	auto pool            = allocator.alloc< PointPool*          >(sizeof(PointPool));
	auto work            = allocator.alloc< PointBatchWorkload* >(sizeof(PointBatchWorkload));
	auto sparseGrid      = allocator.alloc< SparseSampleGrid*   >(sizeof(SparseSampleGrid));
	sparseGrid->keys     = allocator.alloc< uint64_t*           >(2ull * sizeof(uint64_t) * maxGridCapacity);
	sparseGrid->values   = allocator.alloc< uint32_t*           >(4ull * sizeof(uint32_t) * maxGridCapacity);
	
	// GLOBAL STATE
	globals.sparseGrid    = sparseGrid;
	globals.buffer        = buffer;
	globals.input_points  = input_points;
	globals.points        = points;
	globals.lines         = lines;
	globals.triangles     = triangles;
	globals.state         = state;
	globals.allocator     = &allocator;
	globals.octree        = octree;
	globals.pool          = pool;
	dbg = 0;

	{// INIT SPARSE GRID

		vec3 bbmin = vec3(state->metadata.min_x, state->metadata.min_y, state->metadata.min_z);
		vec3 bbmax = vec3(state->metadata.max_x, state->metadata.max_y, state->metadata.max_z);
		vec3 size = bbmax - bbmin;
		float longest = max(max(size.x, size.y), size.z);
		vec3 cubeMax = bbmin + longest;

		sparseGrid->max_capacity = maxGridCapacity;
		sparseGrid->capacity     = maxGridCapacity;
		sparseGrid->min          = bbmin;
		sparseGrid->max          = cubeMax;
		sparseGrid->size         = sparseGrid->max - sparseGrid->min;
	}


	// int gridSize = 256;
	int gridSize = state->guiItems[0].value;
	
	// INIT PRIMITIVE BUFFERS
	points->instanceCount    = 1;
	lines->instanceCount     = 1;
	triangles->instanceCount = 1;

	grid.sync();

	if(isFirstThread()){
		drawBox(triangles, vec3(5.0, 0.0, 0.0), vec3(10.0,  0.1,  0.1), 0xff0000ff);
		drawBox(triangles, vec3(0.0, 5.0, 0.0), vec3( 0.1, 10.0,  0.1), 0xff00ff00);
		drawBox(triangles, vec3(0.0, 0.0, 5.0), vec3( 0.1,  0.1, 10.0), 0xffff0000);
	}

	grid.sync();
	
	{// INIT OCTREE
		vec3 min(state->metadata.min_x, state->metadata.min_y, state->metadata.min_z);
		vec3 max(state->metadata.max_x, state->metadata.max_y, state->metadata.max_z);
		vec3 boxSize = max - min;
		float cubeSize = boxSize.longestAxis();
		vec3 size(cubeSize, cubeSize, cubeSize);
		min = vec3(0.0, 0.0, 0.0);
		max = size;

		Node root;
		root.index = 'r';
		root.id = 0;
		root.boundingBox.min = min;
		root.boundingBox.max = max;
		root.name[0] = 'r';

		octree->min = min;
		octree->max = max;
		octree->size = vec3(cubeSize, cubeSize, cubeSize);
		octree->nodes[0] = root;
		octree->numNodes = 1;
		octree->rootIndex = 0;
		octree->gridSize = gridSize;
		octree->fGridSize = float(gridSize);
	}

	{ // DBG - CREATE CHILD NODES AS WELL
		Node* root = &octree->nodes[0];
		root->childrenIndex = 1;
		root->childrenMask = 0xFF;

		for(int i = 0; i < 8; i++){
			Node child;
			child.index = i;
			child.id = 1 + i;
			child.level = root->level + 1;
			child.boundingBox = computeChildAABB(root->boundingBox, i);
			child.name[0] = root->name[0];
			child.name[1] = child.index;
			
			octree->nodes[1 + i] = child;
		}
		octree->numNodes = 9;
	}

	// INIT POINT POOL
	parallelIterator(0, NUM_BATCHES, [pool](int index){
		pool->batchIndices[index] = index + 1;
	});
	pool->currentBatch = 0;
	
	
	if(grid.thread_rank() == 0){
		drawBoundingBox(lines, octree->size * 0.5f, octree->size, 0x0000ff00);
	}

	grid.sync();

	Timer::timestamp("initialized");

	auto timer = Timer::start("main_topDownSampling");
	main_topDownSampling(state, work, octree);
	Timer::stop(timer);
	// main_sparseGridSampling();

	Timer::timestamp("end");

	if(grid.thread_rank() == 0){
		Timer::instance->t_end = clock64();
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(Timer::instance->t_end_nano));
	}

	grid.sync();

	Timer::print();

	PRINT("STATS\n");
	PRINT("allocated nodes:   %6i \n", octree->numNodes);
	PRINT("allocated batches: %6i \n", pool->currentBatch);
	PRINT("=============================================\n");

}


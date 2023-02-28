
#pragma once

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define FALSE 0
#define TRUE 1

#define LARGE_FLOAT 123000000000.0
#define LARGE_INT   123000000


typedef unsigned int uint32_t;
typedef int int32_t;
typedef char int8_t;
typedef unsigned char uint8_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

#define Infinity 0x7f800000

constexpr uint32_t POINTS_CAPACITY = 10'000'000;

struct Point{
	float x;
	float y;
	float z;
	unsigned int color;
};

struct Voxel{
	float x;
	float y;
	float z;
	unsigned int color;
	float size;
};

struct Points{
	unsigned int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	Point points[POINTS_CAPACITY];
};

struct Lines{
	unsigned int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	Point vertices[10'000'000];
};

struct Boxes{
	uint32_t count;
	float3 positions[1'000'000];
	uint32_t colors[1'000'000];
	float3 sizes[1'000'000];
};

struct Triangles{
	unsigned int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	Point vertices[10'000'000];
};

struct Voxels{
	unsigned int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	Voxel voxels[10'000'000];
};

int test();

struct vec3{

	float x, y, z;

	vec3();
	vec3(float x, float y, float z);

	float dot(vec3);
	float length();
	float longestAxis();

	vec3 normalize();

	friend vec3 operator - (const vec3 &vector, float scalar);
	friend vec3 operator - (const vec3 &vector, vec3 other);
	friend vec3 operator + (const vec3 &vector, vec3 other);
	friend vec3 operator + (const vec3 &vector, float scalar);
	friend vec3 operator / (const vec3 &vector, float scalar);
	friend vec3 operator * (const vec3 &vector, float scalar);
};

struct Box3{
	vec3 min = {Infinity, Infinity, Infinity};
	vec3 max = {-Infinity, -Infinity, Infinity};

	Box3(){}

	Box3(vec3 min, vec3 max){
		this->min = min;
		this->max = max;
	}

	vec3 center();
	vec3 size();
};

struct TimerItem{
	const char* label;
	int64_t start;
	int64_t end;

	void stop();
};

struct Timer{

	struct Timestamp{
		const char* label;
		int64_t time;
	};

	int64_t t_start;
	int64_t t_end;
	int64_t t_start_nano;
	int64_t t_end_nano;

	static const int MAX_MEASUREMENTS = 1'000;

	TimerItem measurements[MAX_MEASUREMENTS];
	int numMeasurements = 0;

	static Timer* instance;

	static void print();

	static TimerItem start(const char* label);
	static void stop(TimerItem item);

};


float strlen(const char* string);
// float clamp(float value, float min, float max);
bool isFirstThread();

// template<typename T>
// void clearBuffer(T* buffer, int offset, int count, T value);

// template<typename T>
// void clearBuffer(T* buffer, int offset, int count, T value){
// 	int totalThreadCount = blockDim.x * gridDim.x;

// 	int itemsPerThread = count / totalThreadCount;
// 	if((count % totalThreadCount) != 0){
// 		itemsPerThread++;
// 	}

// 	int itemsPerGroup = itemsPerThread * blockDim.x;
// 	int group_first = blockIdx.x * itemsPerGroup;
// 	// int thread_first = group_first + itemsPerThread * threadIdx.x;

// 	for(int i = 0; i < itemsPerThread; i++){
// 		int index = group_first + i * blockDim.x + threadIdx.x;

// 		if(index >= offset + count){
// 			continue;
// 		}

// 		buffer[index] = value;
// 	}
// }

template<typename T, typename V>
void clearBuffer(T* buffer, uint64_t byteOffset, uint64_t byteSize, V value){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint64_t totalThreadCount = blockDim.x * gridDim.x;
	uint8_t v8value = value;
	uint32_t v32value = v8value | (v8value << 8) | (v8value << 16) | (v8value << 24);
	uint32_t* v32buffer = reinterpret_cast<uint32_t*>(buffer + byteOffset);

	uint64_t v32Size = byteSize / 4ull;
	uint64_t v32ItemsPerThread = v32Size / totalThreadCount;

	if((byteSize % 4ull) != 0){
		if(isFirstThread()){
			__trap();
			//printf("WARNING: clear should be used with sizes multiple of 4. size is %lli\n", byteSize);
		}
	}
	if((byteOffset % 4ull) != 0){
		if(isFirstThread()){
			__trap();
			//printf("WARNING: clear should be used with offsets multiple of 4. offset is %lli\n", byteOffset);
		}
	}

	if((v32Size % totalThreadCount) != 0){
		v32ItemsPerThread++;
	}

	//grid.sync();

	// if(isFirstThread()){
	// 	printf("byteSize: %i \n", byteSize);
	// 	printf("v32ItemsPerThread: %i \n",  v32ItemsPerThread);
	// 	printf("v32Size: %i \n",  v32Size);
	// 	printf("totalThreadCount: %i \n",  totalThreadCount);
	// }

	{
		uint64_t v32ItemsPerBlock = v32ItemsPerThread * block.num_threads();
		uint64_t blockFirst = blockIdx.x * v32ItemsPerBlock;

		for(int i = 0; i < v32ItemsPerThread; i++){

			uint32_t v32Index = blockFirst + i * block.num_threads() + threadIdx.x;

			if(v32Index < v32Size){
				v32buffer[v32Index] = v32value;

				// printf("v32buffer[%i] = %i; \n", v32Index, v32value);
			}

		}

		//grid.sync();
	}

	// V* vbuffer = reinterpret_cast<V*>(buffer);

	// grid.sync();
}

void drawPoint(Points* points, vec3 pos, uint32_t color);
void drawLine(Lines* lines, vec3 start, vec3 end, uint32_t color);
void drawTriangle(Triangles* triangles, vec3 p1, vec3 p2, vec3 p3, uint32_t color);
void drawBoundingBox(Lines* lines, vec3 pos, vec3 size, uint32_t color);
void drawBox(Triangles* triangles, vec3 pos, vec3 size, uint32_t color);
void drawVoxel(Voxels* voxels, vec3 pos, float size, uint32_t color);

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

#define processRange parallelIterator

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

void printNumber(int64_t number, int leftPad);

struct Allocator{

	uint8_t* buffer = nullptr;
	int64_t offset = 0;

	template<class T>
	Allocator(T buffer){
		this->buffer = reinterpret_cast<uint8_t*>(buffer);
		this->offset = 0;
	}

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
		
		this->offset = newOffset;

		return ptr;
	}

	template<class T>
	T alloc(int64_t size, const char* label){

		// if(isFirstThread()){
		// 	printf("offset: ");
		// 	printNumber(offset, 13);
		// 	printf(", allocating: ");
		// 	printNumber(size, 13);
		// 	printf(", label: %s \n", label);
		// }

		auto ptr = reinterpret_cast<T>(buffer + offset);

		int64_t newOffset = offset + size;
		
		// make allocated buffer location 16-byte aligned to avoid 
		// potential problems with bad alignments
		int64_t remainder = (newOffset % 16ll);

		if(remainder != 0ll){
			newOffset = (newOffset - remainder) + 16ll;
		}
		
		this->offset = newOffset;

		return ptr;
	}

};

inline uint64_t nanotime(){
	uint64_t value;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));

	return value;
}

template<class T>
struct Array{
	uint64_t size;
	T* elements;

	T& operator[](uint32_t index){
		return elements[index];
	}
};
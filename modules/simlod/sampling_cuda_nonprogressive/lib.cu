
#include <cooperative_groups.h>

#include "lib.h.cu"

namespace cg = cooperative_groups;

int test(){
	return 123;
}

vec3::vec3(){
	this->x = 0.0;
	this->y = 0.0;
	this->z = 0.0;
}

vec3::vec3(float x, float y, float z){
	this->x = x;
	this->y = y;
	this->z = z;
}

vec3 vec3::normalize(){
	float length = this->length();

	vec3 result = {
		this->x / length,
		this->y / length,
		this->z / length
	};

	return result;
}

float vec3::dot(vec3 b){
	vec3& a = *this;
	
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

float vec3::length(){
	return sqrtf(x * x + y * y + z * z);
}

float vec3::longestAxis(){
	return fmaxf(fmaxf(x, y), z);
}

vec3 operator-(const vec3 &vector, float scalar){
	vec3 result(
		vector.x - scalar,
		vector.y - scalar,
		vector.z - scalar
	);

	return result;
}

vec3 operator-(const vec3 &vector, vec3 other){
	vec3 result(
		vector.x - other.x,
		vector.y - other.y,
		vector.z - other.z
	);

	return result;
}

vec3 operator+(const vec3 &vector, vec3 other){
	vec3 result(
		vector.x + other.x,
		vector.y + other.y,
		vector.z + other.z
	);

	return result;
}

vec3 operator + (const vec3 &vector, float scalar){
	vec3 result(
		vector.x + scalar,
		vector.y + scalar,
		vector.z + scalar
	);

	return result;
}

vec3 operator / (const vec3 &vector, float scalar){
	vec3 result(
		vector.x / scalar,
		vector.y / scalar,
		vector.z / scalar
	);

	return result;
}

vec3 operator * (const vec3 &vector, float scalar){
	vec3 result(
		vector.x * scalar,
		vector.y * scalar,
		vector.z * scalar
	);

	return result;
}




vec3 Box3::center(){
	return (min + max) / 2.0;
}

vec3 Box3::size(){
	return max - min;
}




Timer* Timer::instance = nullptr;

void TimerItem::stop(){
	Timer::stop(*this);
}

// NOTE: implicitly syncs all(!) threads
void Timer::print(){

	auto grid = cg::this_grid();

	grid.sync();

	if(grid.thread_rank() != 0)              return;
	if(Timer::instance == nullptr)           return;

	Timer* timer = Timer::instance;

	//323'382'519
	printf("== TIMING == \n");

	// printf("start:    %8lli \n", timer->t_start);
	// printf("end:      %8lli \n", timer->t_end);
	// printf("duration: %8lli \n", timer->t_end - timer->t_start);
	printf("label                                          time       \n");
	printf("----------------------------------------------------------\n");

	int64_t duration_total = timer->t_end - timer->t_start;
	int64_t duration_total_nano = timer->t_end_nano - timer->t_start_nano;

	// version 2: sum up duplicates
	for(int i = 0; i < timer->numMeasurements; i++){
		auto measure = timer->measurements[i];

		int64_t duration = measure.end - measure.start;

		if(duration == 0) continue;

		for(int j = i + 1; j < timer->numMeasurements; j++){
			auto ptr_other_measure = &timer->measurements[j];

			if(measure.label == ptr_other_measure->label){
				// duplicate found, merge and invalidate

				duration += ptr_other_measure->end - ptr_other_measure->start;

				ptr_other_measure->end = 0;
				ptr_other_measure->start = 0;
			}
		}

		int64_t duration_percent = 100.0 * double(duration) / double(duration_total);

		double d_nano = double(duration_total_nano) * double(duration_percent) / 100.0;
		double d_ms = d_nano / 1'000'000.0;


		printf("%-45s         ", measure.label);
		// printf("%10i ", (duration / 1000));
		printf(" %5.1f ms ", d_ms);
		printf("(%3llu %%) \n", duration_percent);
	}

	printf("----------------------------------------------------------\n");
}

TimerItem Timer::start(const char* label){

	auto grid = cg::this_grid();
	grid.sync();

	if(Timer::instance == nullptr){
		Timer::instance = new Timer();
	}

	TimerItem item;
	item.label = label;

	if(grid.thread_rank() == 0){
		item.start = clock64();
	}

	return item;
}

void Timer::stop(TimerItem item){

	auto grid = cg::this_grid();

	grid.sync();

	if(grid.thread_rank() != 0) return;

	if(Timer::instance == nullptr){
		Timer::instance = new Timer();
	}
	
	Timer* timer = Timer::instance;

	if(timer->numMeasurements == Timer::MAX_MEASUREMENTS){
		return;
	}

	item.end = clock64();

	timer->measurements[timer->numMeasurements] = item;
	timer->numMeasurements++;

	if(timer->numMeasurements == Timer::MAX_MEASUREMENTS){
		printf("WARNING: maximum number of time measurements reached (%i) \n", Timer::MAX_MEASUREMENTS);
	}
}


float strlen(const char* string){
	int length = 0;
	int MAX_STRING_LENGTH = 100;

	for(int i = 0; i < MAX_STRING_LENGTH; i++){
		if(string[i] == 0){
			break;
		}

		length++;
	}

	return length;
}

template<typename T>
float clamp(T value, T min, T max){
	// value = fmaxf(value, min);
	// value = fminf(value, max);

	if(value < min) return min;
	if(value > max) return max;

	return value;
}

bool isFirstThread(){
	auto grid = cg::this_grid();
	return grid.thread_rank() == 0;
}

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

void drawPoint(Points* points, vec3 pos, uint32_t color){
	Point point;
	point.x = pos.x;
	point.y = pos.y;
	point.z = pos.z;
	point.color = color;

	int pointIndex = atomicAdd(&points->count, 1);

	if(pointIndex < POINTS_CAPACITY){
		points->points[pointIndex] = point;
	}else{
		points->count = POINTS_CAPACITY - 1;
	}

	// { // DEBUG
	// 	if(pointIndex > 7'000'000){
	// 		point.x = point.x * 100000.0;
	// 		point.z = point.z * 100000.0;
	// 		point.y = point.y * 100000.0;
	// 	}else{

	// 	}

	// 	points->points[pointIndex] = point;
	// }
}

void drawLine(Lines* lines, vec3 start, vec3 end, uint32_t color){

	Point pStart;
	pStart.x = start.x;
	pStart.y = start.y;
	pStart.z = start.z;
	pStart.color = color;

	Point pEnd;
	pEnd.x = end.x;
	pEnd.y = end.y;
	pEnd.z = end.z;
	pEnd.color = color;

	int index = atomicAdd(&lines->count, 2);
	lines->vertices[index + 0] = pStart;
	lines->vertices[index + 1] = pEnd;

}

void drawTriangle(Triangles* triangles, vec3 p1, vec3 p2, vec3 p3, uint32_t color){

	Point pt1;
	pt1.x = p1.x;
	pt1.y = p1.y;
	pt1.z = p1.z;
	pt1.color = color;

	Point pt2;
	pt2.x = p2.x;
	pt2.y = p2.y;
	pt2.z = p2.z;
	pt2.color = color;

	Point pt3;
	pt3.x = p3.x;
	pt3.y = p3.y;
	pt3.z = p3.z;
	pt3.color = color;

	int index = atomicAdd(&triangles->count, 3);
	triangles->vertices[index + 0] = pt1;
	triangles->vertices[index + 1] = pt2;
	triangles->vertices[index + 2] = pt3;

}

void drawQuad(Triangles* triangles, vec3 p1, vec3 p2, vec3 p3, vec3 p4, uint32_t color){

	Point pt1;
	pt1.x = p1.x;
	pt1.y = p1.y;
	pt1.z = p1.z;
	pt1.color = color;

	Point pt2;
	pt2.x = p2.x;
	pt2.y = p2.y;
	pt2.z = p2.z;
	pt2.color = color;

	Point pt3;
	pt3.x = p3.x;
	pt3.y = p3.y;
	pt3.z = p3.z;
	pt3.color = color;

	Point pt4;
	pt4.x = p4.x;
	pt4.y = p4.y;
	pt4.z = p4.z;
	pt4.color = color;

	int index = atomicAdd(&triangles->count, 6);
	triangles->vertices[index + 0] = pt1;
	triangles->vertices[index + 1] = pt2;
	triangles->vertices[index + 2] = pt3;
	triangles->vertices[index + 3] = pt1;
	triangles->vertices[index + 4] = pt3;
	triangles->vertices[index + 5] = pt4;

}

void drawBoundingBox(Lines* lines, vec3 pos, vec3 size, uint32_t color){

	vec3 min = pos - size / 2.0;
	vec3 max = pos + size / 2.0;

	// BOTTOM
	drawLine(lines, vec3(min.x, min.y, min.z), vec3(max.x, min.y, min.z), color);
	drawLine(lines, vec3(max.x, min.y, min.z), vec3(max.x, max.y, min.z), color);
	drawLine(lines, vec3(max.x, max.y, min.z), vec3(min.x, max.y, min.z), color);
	drawLine(lines, vec3(min.x, max.y, min.z), vec3(min.x, min.y, min.z), color);

	// TOP
	drawLine(lines, vec3(min.x, min.y, max.z), vec3(max.x, min.y, max.z), color);
	drawLine(lines, vec3(max.x, min.y, max.z), vec3(max.x, max.y, max.z), color);
	drawLine(lines, vec3(max.x, max.y, max.z), vec3(min.x, max.y, max.z), color);
	drawLine(lines, vec3(min.x, max.y, max.z), vec3(min.x, min.y, max.z), color);

	// BOTTOM TO TOP
	drawLine(lines, vec3(min.x, min.y, min.z), vec3(min.x, min.y, max.z), color);
	drawLine(lines, vec3(max.x, min.y, min.z), vec3(max.x, min.y, max.z), color);
	drawLine(lines, vec3(max.x, max.y, min.z), vec3(max.x, max.y, max.z), color);
	drawLine(lines, vec3(min.x, max.y, min.z), vec3(min.x, max.y, max.z), color);

}

void drawBox(Triangles* triangles, vec3 pos, vec3 size, uint32_t color){

	vec3 min = pos - size / 2.0;
	vec3 max = pos + size / 2.0;

	// BOTTOM
	// drawTriangle(triangles, vec3(min.x, min.y, min.z), vec3(max.x, min.y, min.z), vec3(max.x, max.y, min.z), color);
	// drawTriangle(triangles, vec3(min.x, min.y, min.z), vec3(max.x, max.y, min.z), vec3(min.x, max.y, min.z), color);
	drawQuad(triangles, vec3(min.x, min.y, min.z), vec3(max.x, min.y, min.z), vec3(max.x, max.y, min.z), vec3(min.x, max.y, min.z), color);

	// TOP
	// drawTriangle(triangles, vec3(min.x, min.y, max.z), vec3(max.x, min.y, max.z), vec3(max.x, max.y, max.z), color);
	// drawTriangle(triangles, vec3(min.x, min.y, max.z), vec3(max.x, max.y, max.z), vec3(min.x, max.y, max.z), color);
	drawQuad(triangles, vec3(min.x, min.y, max.z), vec3(max.x, min.y, max.z), vec3(max.x, max.y, max.z), vec3(min.x, max.y, max.z), color);

	// RIGHT
	// drawTriangle(triangles, vec3(min.x, min.y, min.z), vec3(max.x, min.y, min.z), vec3(max.x, min.y, max.z), color);
	// drawTriangle(triangles, vec3(min.x, min.y, min.z), vec3(max.x, min.y, max.z), vec3(min.x, min.y, max.z), color);
	drawQuad(triangles, vec3(min.x, min.y, min.z), vec3(max.x, min.y, min.z), vec3(max.x, min.y, max.z), vec3(min.x, min.y, max.z), color);

	// LEFT
	// drawTriangle(triangles, vec3(min.x, max.y, min.z), vec3(max.x, max.y, min.z), vec3(max.x, max.y, max.z), color);
	// drawTriangle(triangles, vec3(min.x, max.y, min.z), vec3(max.x, max.y, max.z), vec3(min.x, max.y, max.z), color);
	drawQuad(triangles, vec3(min.x, max.y, min.z), vec3(max.x, max.y, min.z), vec3(max.x, max.y, max.z), vec3(min.x, max.y, max.z), color);

	// FONT
	// drawTriangle(triangles, vec3(min.x, min.y, min.z), vec3(min.x, max.y, min.z), vec3(min.x, max.y, max.z), color);
	// drawTriangle(triangles, vec3(min.x, min.y, min.z), vec3(min.x, max.y, max.z), vec3(min.x, min.y, max.z), color);
	drawQuad(triangles, vec3(min.x, min.y, min.z), vec3(min.x, max.y, min.z), vec3(min.x, max.y, max.z), vec3(min.x, min.y, max.z), color);

	// BACK
	// drawTriangle(triangles, vec3(max.x, min.y, min.z), vec3(max.x, max.y, min.z), vec3(max.x, max.y, max.z), color);
	// drawTriangle(triangles, vec3(max.x, min.y, min.z), vec3(max.x, max.y, max.z), vec3(max.x, min.y, max.z), color);
	drawQuad(triangles, vec3(max.x, min.y, min.z), vec3(max.x, max.y, min.z), vec3(max.x, max.y, max.z), vec3(max.x, min.y, max.z), color);

}

void drawVoxel(Voxels* voxels, vec3 pos, float size, uint32_t color){
	Voxel voxel;
	voxel.x = pos.x;
	voxel.y = pos.y;
	voxel.z = pos.z;
	voxel.color = color;
	voxel.size = size;

	int verticesPerBox = 6 * 2 * 3;
	int voxelIndex = atomicAdd(&voxels->count, verticesPerBox) / verticesPerBox;
	voxels->voxels[voxelIndex] = voxel;
}


void printNumber(int64_t number, int leftPad = 0){
	int digits = ceil(log10(double(number)));
	digits = (digits > 0) ? digits : 1;
	int numSeparators = (digits - 1) / 3;
	int stringSize = digits + numSeparators;

	// printf("numSeparators: %i, stringSize: %i \n", numSeparators, stringSize);


	for(int i = leftPad; i > stringSize; i--){
		printf(" ");
	}

	for(int digit = digits - 1; digit >= 0; digit--){

		int64_t a = pow(10.0, double(digit) + 1);
		int64_t b = pow(10.0, double(digit));

		int64_t tmp = number / a;
		int64_t tmp2 = number - (a * tmp);
		int64_t current = tmp2 / b;

		printf("%lli", current);

		if((digit % 3) == 0 && digit > 0){
			printf("'");
		}

	}
}
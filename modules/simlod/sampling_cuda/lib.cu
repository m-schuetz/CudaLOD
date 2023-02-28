
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

// NOTE: implicitly syncs all(!) threads
void Timer::timestamp(const char* label){

	auto grid = cg::this_grid();

	grid.sync();

	if(grid.thread_rank() != 0) return;

	if(Timer::instance == nullptr){
		Timer::instance = new Timer();
	}


	Timestamp timestamp;
	timestamp.label = label;
	timestamp.time = clock();

	Timer* timer = Timer::instance;
	timer->timestamps[timer->numTimestamps] = timestamp;
	timer->numTimestamps++;
}

// NOTE: implicitly syncs all(!) threads
void Timer::print(){

	auto grid = cg::this_grid();

	grid.sync();

	if(grid.thread_rank() != 0)              return;
	if(Timer::instance == nullptr)           return;
	if(Timer::instance->numTimestamps == 0)  return;

	Timer* timer = Timer::instance;

	//323'382'519
	printf("== TIMING == \n");

	printf("start:    %8lli \n", timer->t_start);
	printf("end:      %8lli \n", timer->t_end);
	printf("duration: %8lli \n", timer->t_end - timer->t_start);
	printf("label                        time           duration\n");
	printf("------------------------------------------\n");

	// for(int i = 0; i < timer->numTimestamps; i++){
	// 	auto timestamp = timer->timestamps[i];

	// 	int64_t duration = timestamp.time - (timer->timestamps[0].time);
	// 	int64_t duration_rel = (i == 0) ? 0 : (timestamp.time - (timer->timestamps[i - 1].time));
	// 	int64_t duration_total = timer->timestamps[timer->numTimestamps - 1].time - timer->timestamps[0].time;
	// 	int64_t duration_rel_percent = 100.0 * double(duration_rel) / double(duration_total);

	// 	printf("%-25s %7i ", timestamp.label, (duration / 1000), (duration_rel / 1000));
	// 	printf("%10i ", (duration_rel / 1000));
	// 	printf("(%3i %%) \n", duration_rel_percent);
	// }

	// printf("---------------\n");

	int64_t duration_total = timer->t_end - timer->t_start;
	int64_t duration_total_nano = timer->t_end_nano - timer->t_start_nano;
	// int64_t t_duration_nano = (t_end_nano - t_start_nano);
	// int64_t t_duration_ms = t_duration_nano / 1'000'000;

	// version 1: print each separately
	// for(int i = 0; i < timer->numMeasurements; i++){
	// 	auto measure = timer->measurements[i];

	// 	int64_t duration = measure.end - measure.start;
	// 	int64_t duration_percent = 100.0 * double(duration) / double(duration_total);

	// 	printf("%-25s         ", measure.label);
	// 	printf("%10i ", (duration / 1000));
	// 	printf("(%3i %%) \n", duration_percent);
	// }

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


		printf("%-25s         ", measure.label);
		printf("%10i ", (duration / 1000));
		printf("(%3i %%) ", duration_percent);
		printf(" %4i ms \n", int(d_ms));
	}

	printf("------------------------------------------\n");
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

	item.end = clock64();

	Timer* timer = Timer::instance;
	timer->measurements[timer->numMeasurements] = item;
	timer->numMeasurements++;
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

float clamp(float value, float min, float max){
	value = fmaxf(value, min);
	value = fminf(value, max);

	return value;
}

bool isFirstThread(){
	auto grid = cg::this_grid();
	return grid.thread_rank() == 0;
}

void clearBuffer(int* buffer, int offset, int count, int value){
	int totalThreadCount = blockDim.x * gridDim.x;

	int itemsPerThread = count / totalThreadCount;
	if((count % totalThreadCount) != 0){
		itemsPerThread++;
	}

	int itemsPerGroup = itemsPerThread * blockDim.x;
	int group_first = blockIdx.x * itemsPerGroup;
	// int thread_first = group_first + itemsPerThread * threadIdx.x;

	for(int i = 0; i < itemsPerThread; i++){
		int index = group_first + i * blockDim.x + threadIdx.x;

		if(index >= offset + count){
			continue;
		}

		buffer[index] = value;
	}

}

void drawPoint(Points* points, vec3 pos, uint32_t color){
	Point point;
	point.x = pos.x;
	point.y = pos.y;
	point.z = pos.z;
	point.color = color;

	int pointIndex = atomicAdd(&points->count, 1);
	points->points[pointIndex] = point;

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
	drawTriangle(triangles, vec3(min.x, min.y, min.z), vec3(max.x, min.y, min.z), vec3(max.x, max.y, min.z), color);
	drawTriangle(triangles, vec3(min.x, min.y, min.z), vec3(max.x, max.y, min.z), vec3(min.x, max.y, min.z), color);

	// TOP
	drawTriangle(triangles, vec3(min.x, min.y, max.z), vec3(max.x, min.y, max.z), vec3(max.x, max.y, max.z), color);
	drawTriangle(triangles, vec3(min.x, min.y, max.z), vec3(max.x, max.y, max.z), vec3(min.x, max.y, max.z), color);

	// RIGHT
	drawTriangle(triangles, vec3(min.x, min.y, min.z), vec3(max.x, min.y, min.z), vec3(max.x, min.y, max.z), color);
	drawTriangle(triangles, vec3(min.x, min.y, min.z), vec3(max.x, min.y, max.z), vec3(min.x, min.y, max.z), color);

	// LEFT
	drawTriangle(triangles, vec3(min.x, max.y, min.z), vec3(max.x, max.y, min.z), vec3(max.x, max.y, max.z), color);
	drawTriangle(triangles, vec3(min.x, max.y, min.z), vec3(max.x, max.y, max.z), vec3(min.x, max.y, max.z), color);

	// FONT
	drawTriangle(triangles, vec3(min.x, min.y, min.z), vec3(min.x, max.y, min.z), vec3(min.x, max.y, max.z), color);
	drawTriangle(triangles, vec3(min.x, min.y, min.z), vec3(min.x, max.y, max.z), vec3(min.x, min.y, max.z), color);

	// BACK
	drawTriangle(triangles, vec3(max.x, min.y, min.z), vec3(max.x, max.y, min.z), vec3(max.x, max.y, max.z), color);
	drawTriangle(triangles, vec3(max.x, min.y, min.z), vec3(max.x, max.y, max.z), vec3(max.x, min.y, max.z), color);

}

// int syncGlobalAuto(uint32_t* semaphores, Triangles* triangles, int semaphoreIndex){
	
// 	uint32_t totalThreadCount = blockDim.x * gridDim.x;
// 	int target = totalThreadCount;

// 	int i = 0; 
// 	while(atomicAdd(&semaphores[semaphoreIndex], 0) != target){

// 		// syncthreads causes complete block?!
// 		// __syncthreads();
// 		__nanosleep(10);

// 		if(i > 10000){
// 			drawBox(triangles, vec3(0, 0, 0), vec3(1, 1, 0.1), 0xff0000ff);
// 			// TODO draw red box as error hint
// 			break;
// 		}

// 		i++;
// 	}

// 	return semaphoreIndex + 1;
// }

void syncGlobalAuto(uint32_t* semaphores, Triangles* triangles){
	
	uint32_t totalThreadCount = blockDim.x * gridDim.x;
	// int target = totalThreadCount;

	int sem = atomicAdd_system(&semaphores[0], 1);
	int target = (1 + (sem / totalThreadCount)) * totalThreadCount;

	if(isFirstThread()) printf("target: %i, sem: %i \n", target, sem);

	int i = 0; 
	while(atomicAdd_system(&semaphores[0], 0) != target){

		// syncthreads causes complete block?!
		// __syncthreads();
		__nanosleep(10);

		if(i > 10000){
			drawBox(triangles, vec3(0, 0, 0), vec3(1, 1, 0.1), 0xff0000ff);
			// TODO draw red box as error hint
			break;
		}

		i++;
	}

	semaphores[0] = target;

}

void syncGlobal(uint32_t* semaphores, Triangles* triangles, int semaphoreIndex){
	uint32_t totalThreadCount = blockDim.x * gridDim.x;

	atomicAdd(&semaphores[semaphoreIndex], 1);

	int i = 0; 
	while(atomicAdd(&semaphores[semaphoreIndex], 0) != totalThreadCount){

		// syncthreads causes complete block?!
		// __syncthreads();
		__nanosleep(10);

		if(i > 10000){
			drawBox(triangles, vec3(0, 0, 0), vec3(1, 1, 0.1), 0xff0000ff);
			// TODO draw red box as error hint
			return;
		}

		i++;
	}

}


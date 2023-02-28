#version 450

#extension GL_NV_shader_atomic_int64 : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_NV_shader_atomic_float : require
#extension GL_ARB_shader_clock : require

#define Infinity (1.0 / 0.0)
#define MAX_DEPTH 10
#define MAX_POINTS_PER_NODE 100000
#define GRID_SIZE 256

layout(local_size_x = 128, local_size_y = 1) in;

struct Point{
	float x;
	float y;
	float z;
	uint32_t color;
};

struct Node{
	int level;
	int count;
	// pointers/indices to child nodes. -1 if empty child
	int children[8]; 
};

layout (std430, binding =  0) buffer abc_0 { Point ssPoints[]; };

layout (std430, binding =  5) buffer abc_5 { uint32_t ssBuffer[]; };

layout (std430, binding = 10) buffer abc_10 { 
	uint  count;
	uint  instanceCount;
	uint  first;
	uint  baseInstance;
	Point points[];
} out_points;

layout (std430, binding = 11) buffer abc_11 { 
	uint  count;
	uint  instanceCount;
	uint  first;
	uint  baseInstance;
	Point vertices[];
} out_lines;

layout(std140, binding = 20) uniform UniformData{
	int numPoints;
	float min_x;
	float min_y;
	float min_z;
	float max_x;
	float max_y;
	float max_z;
} uniforms;

layout (std430, binding = 21) coherent buffer abc_21 { 
	uint32_t counter_0;
	uint32_t counter_1;
	uint32_t counter_2;
	uint32_t counter_3;
	uint32_t semaphores[8];
} state;

layout (std430, binding = 30) buffer abc_20 { 
	uint32_t value;
	bool enabled;
	uint32_t numPoints;
} debug;



void drawPoint(vec3 pos, vec3 color, int size){
	Point point;
	point.x = pos.x;
	point.y = pos.y;
	point.z = pos.z;

	point.color = 
		(int(255.0 * color.x) <<  0) |
		(int(255.0 * color.y) <<  8) |
		(int(255.0 * color.z) << 16) |
		(size << 24) ;


	uint pointIndex = atomicAdd(out_points.count, 1);

	out_points.points[pointIndex] = point;
}

void drawLine(vec3 start, vec3 end, vec3 color){

	uint ucolor = 
		(int(255.0 * color.x) <<  0) |
		(int(255.0 * color.y) <<  8) |
		(int(255.0 * color.z) << 16);

	Point pStart;
	pStart.x = start.x;
	pStart.y = start.y;
	pStart.z = start.z;
	pStart.color = ucolor;

	Point pEnd;
	pEnd.x = end.x;
	pEnd.y = end.y;
	pEnd.z = end.z;
	pEnd.color = ucolor;

	uint lineIndex = atomicAdd(out_lines.count, 2);

	// atomicMax(debug.value, out_lines.count);

	out_lines.vertices[lineIndex + 0] = pStart;
	out_lines.vertices[lineIndex + 1] = pEnd;
}

void drawBox(vec3 pos, vec3 size, vec3 color){

	vec3 min = pos - size / 2.0;
	vec3 max = pos + size / 2.0;

	// bottom
	drawLine(vec3(min.x, min.y, min.z), vec3(max.x, min.y, min.z), color);
	drawLine(vec3(max.x, min.y, min.z), vec3(max.x, max.y, min.z), color);
	drawLine(vec3(max.x, max.y, min.z), vec3(min.x, max.y, min.z), color);
	drawLine(vec3(min.x, max.y, min.z), vec3(min.x, min.y, min.z), color);

	// top
	drawLine(vec3(min.x, min.y, max.z), vec3(max.x, min.y, max.z), color);
	drawLine(vec3(max.x, min.y, max.z), vec3(max.x, max.y, max.z), color);
	drawLine(vec3(max.x, max.y, max.z), vec3(min.x, max.y, max.z), color);
	drawLine(vec3(min.x, max.y, max.z), vec3(min.x, min.y, max.z), color);

	// bottom to top
	drawLine(vec3(min.x, min.y, min.z), vec3(min.x, min.y, max.z), color);
	drawLine(vec3(max.x, min.y, min.z), vec3(max.x, min.y, max.z), color);
	drawLine(vec3(max.x, max.y, min.z), vec3(max.x, max.y, max.z), color);
	drawLine(vec3(min.x, max.y, min.z), vec3(min.x, max.y, max.z), color);
}

// void drawBox(vec3 pos, vec3 size, vec3 color){

// 	float eps = 0.001;

// 	_drawBox(pos, size, color);
// 	_drawBox(pos + eps, size, color);
// 	_drawBox(pos - eps, size, color);
// 	_drawBox(pos + 2.0 * eps, size, color);
// 	_drawBox(pos - 2.0 * eps, size, color);
// }

void drawCircle(vec3 pos, float radius, vec3 color){

	float PI = 3.1415;

	float steps = 50.0;
	for(int i = 0; i < steps; i++){

		float ra = 2.0 * PI * (i / steps);
		float rb = 2.0 * PI * ((i + 1) / steps);

		vec3 start = vec3(cos(ra), sin(ra), 0.0);
		vec3 end = vec3(cos(rb), sin(rb), 0.0);

		drawLine(start, end, color);

	}

}

uint computeItemsPerThread(int itemCount){
	uint totalThreadCount = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

	uint itemsPerThread = itemCount / totalThreadCount;

	if((itemCount % totalThreadCount) != 0){
		itemsPerThread++;
	}

	return itemsPerThread;
}

void clearBuffer_i32(int offset, int count, int value){

	uint totalThreadCount = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

	uint itemsPerThread = count / totalThreadCount;

	if((count % totalThreadCount) != 0){
		itemsPerThread++;
	}

	uint itemsPerGroup = itemsPerThread * gl_WorkGroupSize.x;
	uint group_first = gl_WorkGroupID.x * itemsPerGroup;
	uint thread_first = group_first + itemsPerThread * gl_LocalInvocationID.x;

	for(int i = 0; i < itemsPerThread; i++){
		ssBuffer[thread_first + i] = value;
	}

	// if(gl_WorkGroupID.x == 0 && gl_LocalInvocationID.x == 3){
	// 	debug.value = thread_first;
	// }

}

void init(){
	out_points.instanceCount = 1;
	out_points.first = 0;
	out_points.baseInstance = 0;

	out_lines.instanceCount = 1;
	out_lines.first = 0;
	out_lines.baseInstance = 0;
}

// void main(){

// 	uint workgroupID = gl_WorkGroupID.x;
// 	uint threadID = gl_GlobalInvocationID.x;

// 	init();

// 	// if(threadID == 0){
// 	// 	vec3 min = vec3(uniforms.min_x, uniforms.min_y, uniforms.min_z);
// 	// 	vec3 max = vec3(uniforms.max_x, uniforms.max_y, uniforms.max_z);
// 	// 	vec3 size = max - min;
// 	// 	vec3 pos = size / 2.0;
// 	// 	drawBox(
// 	// 		pos, size, 
// 	// 		vec3(1.0, 0.0, 1.0));
// 	// }
	

// 	// int pointsPerThread = 100;
// 	// uint batchSize = gl_WorkGroupSize.x * pointsPerThread;

// 	// uint wgFirstPoint = workgroupID * batchSize;
// 	// for(int i = 0; i < pointsPerThread; i++){

// 	// 	uint localIndex = i * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
// 	// 	uint globalIndex = wgFirstPoint + localIndex;

// 	// 	Point point = ssPoints[globalIndex * 4];

// 	// 	drawPoint(vec3(point.x, point.y, point.z),
// 	// 		vec3(
// 	// 			(point.color >>  0) & 0xFF, 
// 	// 			(point.color >>  8) & 0xFF, 
// 	// 			(point.color >> 16) & 0xFF
// 	// 		) / 256.0
// 	// 	);

// 	// }

	

	



// }


// void drawStuff(){
// 	uint workgroupID = gl_WorkGroupID.x;
// 	uint threadID = gl_GlobalInvocationID.x;
// 	uint totalThreadCount = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

// 	init();

// 	if(threadID == 0){

// 		vec3 yellow = vec3(1.0, 1.0, 0.0);
// 		vec3 black = vec3(0.0, 0.0, 0.0);
// 		vec3 red = vec3(1.0, 0.0, 0.0);

// 		drawCircle(vec3(0.0, 0.0, 0.0), 1.0, yellow);

// 		drawPoint(vec3(-0.5, 0.1, 0.0), black, 30);
// 		drawPoint(vec3(+0.5, 0.1, 0.0), black, 30);

// 		drawLine(vec3(-0.5, -0.3, 0.0), vec3( 0.0, -0.5, 0.0), yellow);
// 		drawLine(vec3( 0.0, -0.5, 0.0), vec3(+0.5, -0.3, 0.0), yellow);

// 		drawBox(vec3(0.0, 1.0, 0.0), vec3(2.0, 0.2, 0.4), red);
// 		drawBox(vec3(0.0, 1.8, 0.0), vec3(1.5, 1.5, 0.3), red);



// 		drawBox(vec3(0.5, 0.5, 0.5), vec3(1.0, 1.0, 1.0), red);
// 	}
// }


void syncGlobal(int semaphoreIndex){

	uint64_t tStart = clockARB();

	uint totalThreadCount = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

	// all threads add 1 to the semaphore
	atomicAdd(state.semaphores[semaphoreIndex], 1);

	// spin-loop until all threads incremented the semaphore
	int i = 0;
	// while(state.semaphores[semaphoreIndex] != totalThreadCount){
	while(atomicAdd(state.semaphores[semaphoreIndex], 0) != totalThreadCount){
		memoryBarrier();
		memoryBarrierBuffer();

		// avoid deadlocks. should do some special handling.
		if(i > 5000){

			// draw a red box on "timeout"
			drawBox(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), vec3(1.0, 0.0, 0.0));

			break;
		}

		i++;
	}

	uint64_t tEnd = clockARB();
	uint64_t duration = (tEnd - tStart) / 1000ul;

	// if(semaphoreIndex == 01){
	// 	debug.value = uint32_t(duration);
	// }
}

// void main(){

// 	uint workgroupID = gl_WorkGroupID.x;
// 	uint threadID = gl_GlobalInvocationID.x;
// 	uint totalThreadCount = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

// 	init();

// 	// clearBuffer_i32(0, 11000, 0);
// 	clearBuffer_i32(0, 4 * 128 * 128 * 128, 0);

// 	// semaphore_0();
// 	syncGlobal(0);

// 	vec3 boxMin = vec3(uniforms.min_x, uniforms.min_y, uniforms.min_z);
// 	vec3 boxMax = vec3(uniforms.max_x, uniforms.max_y, uniforms.max_z);
// 	vec3 boxSize = boxMax - boxMin;
// 	int gridSize = 128;
// 	float fGridSize = float(gridSize);

// 	{
// 		uint pointsPerThread = computeItemsPerThread(uniforms.numPoints);

// 		uint itemsPerGroup = pointsPerThread * gl_WorkGroupSize.x;
// 		uint group_first = gl_WorkGroupID.x * itemsPerGroup;
// 		uint thread_first = group_first + pointsPerThread * gl_LocalInvocationID.x;

// 		for(int i = 0; i < pointsPerThread; i++){

// 			uint globalIndex = thread_first + i;

// 			if(globalIndex > uniforms.numPoints){
// 				break;
// 			}

// 			Point point = ssPoints[globalIndex];

// 			float fx = fGridSize * (point.x / boxSize.x);
// 			float fy = fGridSize * (point.y / boxSize.y);
// 			float fz = fGridSize * (point.z / boxSize.z);

// 			int ix = int(clamp(fx, 0.0, fGridSize - 1.0));
// 			int iy = int(clamp(fy, 0.0, fGridSize - 1.0));
// 			int iz = int(clamp(fz, 0.0, fGridSize - 1.0));

// 			int voxelIndex = ix + iy * gridSize + iz * gridSize * gridSize;

// 			uint R = (point.color >>  0) & 0xFF;
// 			uint G = (point.color >>  8) & 0xFF;
// 			uint B = (point.color >> 16) & 0xFF;
// 			atomicAdd(ssBuffer[4 * voxelIndex + 0], R);
// 			atomicAdd(ssBuffer[4 * voxelIndex + 1], G);
// 			atomicAdd(ssBuffer[4 * voxelIndex + 2], B);
// 			uint voxelCount = atomicAdd(ssBuffer[4 * voxelIndex + 3], 1);

// 			// if(voxelCount == 0){
// 			// // if((globalIndex % 100) == 0){
// 			// 	Point point = ssPoints[globalIndex];

// 			// 	drawPoint(vec3(point.x, point.y, point.z),
// 			// 		vec3(
// 			// 			(point.color >>  0) & 0xFF, 
// 			// 			(point.color >>  8) & 0xFF, 
// 			// 			(point.color >> 16) & 0xFF
// 			// 		) / 256.0,
// 			// 		10
// 			// 	);
// 			// }


// 		}
// 	}

// 	syncGlobal(1);

// 	{
// 		uint voxelsPerThread = computeItemsPerThread(gridSize * gridSize * gridSize);

// 		uint itemsPerGroup = voxelsPerThread * gl_WorkGroupSize.x;
// 		uint group_first = gl_WorkGroupID.x * itemsPerGroup;
// 		uint thread_first = group_first + voxelsPerThread * gl_LocalInvocationID.x;

// 		for(int i = 0; i < voxelsPerThread; i++){

// 			uint voxelIndex = thread_first + i;
// 			uint ix = voxelIndex % gridSize;
// 			uint iz = voxelIndex / (gridSize * gridSize);
// 			uint iy = (voxelIndex % (gridSize * gridSize)) / gridSize;

// 			float x = (float(ix) / fGridSize) * boxSize.x + 0.5 * boxSize.x / fGridSize;
// 			float y = (float(iy) / fGridSize) * boxSize.y + 0.5 * boxSize.y / fGridSize;
// 			float z = (float(iz) / fGridSize) * boxSize.z + 0.5 * boxSize.z / fGridSize;

// 			uint R = ssBuffer[4 * voxelIndex + 0];
// 			uint G = ssBuffer[4 * voxelIndex + 1];
// 			uint B = ssBuffer[4 * voxelIndex + 2];
// 			uint count = ssBuffer[4 * voxelIndex + 3];

// 			float r = float(R / count) / 256.0;
// 			float g = float(G / count) / 256.0;
// 			float b = float(B / count) / 256.0;

// 			if(count > 0){
// 				drawPoint(vec3(x, y, z), vec3(r, g, b), 10);
// 			}






// 		}

// 	}


// 	// if(threadID == 1){
// 	// 	debug.value = state.semaphores[0];
// 	// }
// }

void projectPointsToVoxelGrid(){
	uint workgroupID = gl_WorkGroupID.x;
	uint threadID = gl_GlobalInvocationID.x;
	uint totalThreadCount = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

	vec3 boxMin = vec3(uniforms.min_x, uniforms.min_y, uniforms.min_z);
	vec3 boxMax = vec3(uniforms.max_x, uniforms.max_y, uniforms.max_z);
	vec3 boxSize = boxMax - boxMin;
	int gridSize = GRID_SIZE;
	float fGridSize = float(gridSize);

	{
		uint pointsPerThread = computeItemsPerThread(uniforms.numPoints);

		uint itemsPerGroup = pointsPerThread * gl_WorkGroupSize.x;
		uint group_first = gl_WorkGroupID.x * itemsPerGroup;
		uint thread_first = group_first + pointsPerThread * gl_LocalInvocationID.x;

		for(int i = 0; i < pointsPerThread; i++){

			uint globalIndex = thread_first + i;

			if(globalIndex > uniforms.numPoints){
				break;
			}

			Point point = ssPoints[globalIndex];

			float fx = fGridSize * (point.x / boxSize.x);
			float fy = fGridSize * (point.y / boxSize.y);
			float fz = fGridSize * (point.z / boxSize.z);

			int ix = int(clamp(fx, 0.0, fGridSize - 1.0));
			int iy = int(clamp(fy, 0.0, fGridSize - 1.0));
			int iz = int(clamp(fz, 0.0, fGridSize - 1.0));

			int voxelIndex = ix + iy * gridSize + iz * gridSize * gridSize;

			uint R = (point.color >>  0) & 0xFF;
			uint G = (point.color >>  8) & 0xFF;
			uint B = (point.color >> 16) & 0xFF;
			atomicAdd(ssBuffer[4 * voxelIndex + 0], R);
			atomicAdd(ssBuffer[4 * voxelIndex + 1], G);
			atomicAdd(ssBuffer[4 * voxelIndex + 2], B);
			uint voxelCount = atomicAdd(ssBuffer[4 * voxelIndex + 3], 1);

			// if(voxelCount == 0){
			// // if((globalIndex % 100) == 0){
			// 	Point point = ssPoints[globalIndex];

			// 	drawPoint(vec3(point.x, point.y, point.z),
			// 		vec3(
			// 			(point.color >>  0) & 0xFF, 
			// 			(point.color >>  8) & 0xFF, 
			// 			(point.color >> 16) & 0xFF
			// 		) / 256.0,
			// 		10
			// 	);
			// }
		}
	}
}

void voxelGridToVertexBuffer(){

	uint workgroupID = gl_WorkGroupID.x;
	uint threadID = gl_GlobalInvocationID.x;
	uint totalThreadCount = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

	vec3 boxMin = vec3(uniforms.min_x, uniforms.min_y, uniforms.min_z);
	vec3 boxMax = vec3(uniforms.max_x, uniforms.max_y, uniforms.max_z);
	vec3 boxSize = boxMax - boxMin;
	int gridSize = GRID_SIZE;
	float fGridSize = float(gridSize);

	uint voxelsPerThread = computeItemsPerThread(gridSize * gridSize * gridSize);

	uint itemsPerGroup = voxelsPerThread * gl_WorkGroupSize.x;
	uint group_first = gl_WorkGroupID.x * itemsPerGroup;
	uint thread_first = group_first + voxelsPerThread * gl_LocalInvocationID.x;

	for(int i = 0; i < voxelsPerThread; i++){

		uint voxelIndex = thread_first + i;
		uint ix = voxelIndex % gridSize;
		uint iz = voxelIndex / (gridSize * gridSize);
		uint iy = (voxelIndex % (gridSize * gridSize)) / gridSize;

		float x = (float(ix) / fGridSize) * boxSize.x + 0.5 * boxSize.x / fGridSize;
		float y = (float(iy) / fGridSize) * boxSize.y + 0.5 * boxSize.y / fGridSize;
		float z = (float(iz) / fGridSize) * boxSize.z + 0.5 * boxSize.z / fGridSize;

		uint R = ssBuffer[4 * voxelIndex + 0];
		uint G = ssBuffer[4 * voxelIndex + 1];
		uint B = ssBuffer[4 * voxelIndex + 2];
		uint count = ssBuffer[4 * voxelIndex + 3];

		float r = float(R / count) / 256.0;
		float g = float(G / count) / 256.0;
		float b = float(B / count) / 256.0;

		if(count > 50){
			drawPoint(vec3(x, y, z), vec3(r, g, b), 8);
		}
	}

}

void sort(int first, int count, ){

}

void main(){

	init();

	// draw point cloud bounding box
	if(gl_GlobalInvocationID.x == 0){
		vec3 min = vec3(uniforms.min_x, uniforms.min_y, uniforms.min_z);
		vec3 max = vec3(uniforms.max_x, uniforms.max_y, uniforms.max_z);
		vec3 size = max - min;
		vec3 pos = size / 2.0;
		drawBox(pos, size, vec3(1.0, 0.0, 1.0));
	}

	uint64_t tStart = clockARB();

	// clear voxel grid
	clearBuffer_i32(0, 4 * GRID_SIZE * GRID_SIZE * GRID_SIZE, 0);
	syncGlobal(0);

	uint64_t tCleared = clockARB();

	projectPointsToVoxelGrid();
	syncGlobal(1);

	uint64_t tProjected = clockARB();

	voxelGridToVertexBuffer();
	syncGlobal(2);

	uint64_t tDone = clockARB();


	uint64_t time = (tCleared - tStart) / 1000ul;
	uint64_t duration_clear = (tCleared - tStart) / 1000ul;
	uint64_t duration_project = (tProjected - tCleared) / 1000ul;
	uint64_t duration_extract = (tDone - tProjected) / 1000ul;

	debug.value = uint32_t(duration_project);

}
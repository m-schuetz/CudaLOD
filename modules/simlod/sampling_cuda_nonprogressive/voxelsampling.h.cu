
#include <cooperative_groups.h>
#include "lib.h.cu"

namespace cg = cooperative_groups;

#define GRID_SIZE 128

void method_first_sample(
	Points* points,
	Lines* lines, 
	Triangles* triangles, 
	Allocator& allocator, 
	Box3 box, 
	Point* input_points, 
	int numPoints
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	uint32_t totalThreadCount = blockDim.x * gridDim.x;

	int gridSize = GRID_SIZE;
	float fGridSize = gridSize;
	int numCells = gridSize * gridSize * gridSize;
	uint32_t* countingGrid = allocator.alloc<uint32_t*>(sizeof(uint32_t) * 4 * numCells);

	parallelIterator(0, 4 * numCells, [countingGrid](int index){
		countingGrid[index] = 0;
	});

	grid.sync();

	vec3 boxSize = box.size();

	auto t_count = Timer::start("count");

	{ // only project to single voxel
		uint32_t pointsPerThread = numPoints / totalThreadCount + 1;
		uint32_t pointsPerBlock = pointsPerThread * block.num_threads();
		uint32_t blockFirst = blockIdx.x * pointsPerBlock;

		for(int i = 0; i < pointsPerThread; i++){

			uint32_t pointIndex = blockFirst + i * block.num_threads() + threadIdx.x;

			if(pointIndex >= numPoints) break;

			Point point = input_points[pointIndex];

			float fx = fGridSize * (point.x) / boxSize.x;
			float fy = fGridSize * (point.y) / boxSize.y;
			float fz = fGridSize * (point.z) / boxSize.z;

			uint32_t ix = clamp(fx, 0.0f, fGridSize - 1.0f);
			uint32_t iy = clamp(fy, 0.0f, fGridSize - 1.0f);
			uint32_t iz = clamp(fz, 0.0f, fGridSize - 1.0f);

			uint32_t voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

			uint32_t R = (point.color >>  0) & 0xff;
			uint32_t G = (point.color >>  8) & 0xff;
			uint32_t B = (point.color >> 16) & 0xff;

			// uint32_t oldCount = atomicAdd(&countingGrid[4 * voxelIndex + 3], 1);
			uint32_t oldCount = atomicCAS(&countingGrid[4 * voxelIndex + 3], 0, 1);

			if(oldCount == 0){
				atomicAdd(&countingGrid[4 * voxelIndex + 0], R);
				atomicAdd(&countingGrid[4 * voxelIndex + 1], G);
				atomicAdd(&countingGrid[4 * voxelIndex + 2], B);
			}
		}
	}

	grid.sync();

	t_count.stop();

	parallelIterator(0, gridSize * gridSize * gridSize, [points, lines, triangles, gridSize, boxSize, countingGrid](int index){
		int ix = index % gridSize;
		int iy = (index % (gridSize * gridSize)) / gridSize;
		int iz = index / (gridSize * gridSize);

		int voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

		int counter = countingGrid[4 * voxelIndex + 3];

		if(counter > 0){
			float x = (float(ix) + 0.5) * boxSize.x / float(gridSize);
			float y = (float(iy) + 0.5) * boxSize.y / float(gridSize);
			float z = (float(iz) + 0.5) * boxSize.z / float(gridSize);
			float cubeSize = boxSize.x / float(gridSize);

			vec3 pos = {x, y, z};
			vec3 size = {cubeSize, cubeSize, cubeSize};
			uint32_t color = 0x000000ff;

			uint32_t R = countingGrid[4 * voxelIndex + 0] / counter;
			uint32_t G = countingGrid[4 * voxelIndex + 1] / counter;
			uint32_t B = countingGrid[4 * voxelIndex + 2] / counter;

			color = R | (G << 8) | (B << 16);

			// drawBoundingBox(lines, pos, size, color);
			drawBox(triangles, pos, size, color);

		}

	});
}

void method_nearest(
	Points* points,
	Lines* lines, 
	Triangles* triangles, 
	Allocator& allocator, 
	Box3 box, 
	Point* input_points, 
	int numPoints
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	uint32_t totalThreadCount = blockDim.x * gridDim.x;

	int gridSize = GRID_SIZE;
	float fGridSize = gridSize;
	int numCells = gridSize * gridSize * gridSize;
	uint32_t* countingGrid = allocator.alloc<uint32_t*>(sizeof(uint32_t) * 4 * numCells);

	parallelIterator(0, 4 * numCells, [countingGrid](int index){
		countingGrid[index] = 0;
	});

	grid.sync();

	vec3 boxSize = box.size();

	auto t_count = Timer::start("count");

	{ // only project to single voxel
		uint32_t pointsPerThread = numPoints / totalThreadCount + 1;
		uint32_t pointsPerBlock = pointsPerThread * block.num_threads();
		uint32_t blockFirst = blockIdx.x * pointsPerBlock;

		for(int i = 0; i < pointsPerThread; i++){

			uint32_t pointIndex = blockFirst + i * block.num_threads() + threadIdx.x;

			if(pointIndex >= numPoints) break;

			Point point = input_points[pointIndex];

			float fx = fGridSize * (point.x) / boxSize.x;
			float fy = fGridSize * (point.y) / boxSize.y;
			float fz = fGridSize * (point.z) / boxSize.z;

			uint32_t ix = clamp(fx, 0.0f, fGridSize - 1.0f);
			uint32_t iy = clamp(fy, 0.0f, fGridSize - 1.0f);
			uint32_t iz = clamp(fz, 0.0f, fGridSize - 1.0f);

			uint32_t voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

			uint32_t R = (point.color >>  0) & 0xff;
			uint32_t G = (point.color >>  8) & 0xff;
			uint32_t B = (point.color >> 16) & 0xff;

			atomicAdd(&countingGrid[4 * voxelIndex + 0], R);
			atomicAdd(&countingGrid[4 * voxelIndex + 1], G);
			atomicAdd(&countingGrid[4 * voxelIndex + 2], B);
			atomicAdd(&countingGrid[4 * voxelIndex + 3], 1);
		}
	}

	grid.sync();

	t_count.stop();

	parallelIterator(0, gridSize * gridSize * gridSize, [points, lines, triangles, gridSize, boxSize, countingGrid](int index){
		int ix = index % gridSize;
		int iy = (index % (gridSize * gridSize)) / gridSize;
		int iz = index / (gridSize * gridSize);

		int voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

		int counter = countingGrid[4 * voxelIndex + 3];

		if(counter > 0){
			float x = (float(ix) + 0.5) * boxSize.x / float(gridSize);
			float y = (float(iy) + 0.5) * boxSize.y / float(gridSize);
			float z = (float(iz) + 0.5) * boxSize.z / float(gridSize);
			float cubeSize = boxSize.x / float(gridSize);

			vec3 pos = {x, y, z};
			vec3 size = {cubeSize, cubeSize, cubeSize};
			uint32_t color = 0x000000ff;

			uint32_t R = countingGrid[4 * voxelIndex + 0] / counter;
			uint32_t G = countingGrid[4 * voxelIndex + 1] / counter;
			uint32_t B = countingGrid[4 * voxelIndex + 2] / counter;

			color = R | (G << 8) | (B << 16);

			// drawBoundingBox(lines, pos, size, color);
			drawBox(triangles, pos, size, color);

		}

	});
}

void method_interpolate(
	Points* points,
	Lines* lines, 
	Triangles* triangles, 
	Allocator& allocator, 
	Box3 box, 
	Point* input_points, 
	int numPoints
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	uint32_t totalThreadCount = blockDim.x * gridDim.x;

	int gridSize = GRID_SIZE;
	float fGridSize = gridSize;
	int numCells = gridSize * gridSize * gridSize;
	uint32_t* flagGrid = allocator.alloc<uint32_t*>(sizeof(uint32_t) * numCells);
	uint32_t* countingGrid = allocator.alloc<uint32_t*>(sizeof(uint32_t) * 4 * numCells);
	
	struct Tracker{
		uint32_t numVoxels;
		uint32_t indices[1'000'000];
	};
	Tracker* tracker = allocator.alloc<Tracker*>(sizeof(Tracker));
	tracker->numVoxels = 0;

	parallelIterator(0, 4 * numCells, [countingGrid](int index){
		countingGrid[index] = 0;
	});

	parallelIterator(0, numCells, [flagGrid](int index){
		flagGrid[index] = 0;
	});

	grid.sync();

	vec3 boxSize = box.size();

	auto t_flag = Timer::start("count");

	{ // create flag grid
		uint32_t pointsPerThread = numPoints / totalThreadCount + 1;
		uint32_t pointsPerBlock = pointsPerThread * block.num_threads();
		uint32_t blockFirst = blockIdx.x * pointsPerBlock;

		for(int i = 0; i < pointsPerThread; i++){

			uint32_t pointIndex = blockFirst + i * block.num_threads() + threadIdx.x;

			if(pointIndex >= numPoints) break;

			Point point = input_points[pointIndex];

			float fx = fGridSize * (point.x) / boxSize.x;
			float fy = fGridSize * (point.y) / boxSize.y;
			float fz = fGridSize * (point.z) / boxSize.z;

			uint32_t ix = clamp(fx, 0.0f, fGridSize - 1.0f);
			uint32_t iy = clamp(fy, 0.0f, fGridSize - 1.0f);
			uint32_t iz = clamp(fz, 0.0f, fGridSize - 1.0f);

			uint32_t voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

			uint32_t R = (point.color >>  0) & 0xff;
			uint32_t G = (point.color >>  8) & 0xff;
			uint32_t B = (point.color >> 16) & 0xff;

			atomicAdd(&flagGrid[voxelIndex], 1);

			// uint32_t oldCount = atomicAdd(&flagGrid[voxelIndex], 1);
			// uint32_t trackIndex = atomicAdd(&tracker->numVoxels, 1);
			// tracker->indices[trackIndex] = voxelIndex;
		}
	}

	// parallelIterator(0, gridSize * gridSize * gridSize, [&](int voxelIndex){
	// 	if(flagGrid[voxelIndex] > 0){
	// 		uint32_t trackIndex = atomicAdd(&tracker->numVoxels, 1);
	// 		tracker->indices[trackIndex] = voxelIndex;
	// 	}
	// });

	t_flag.stop();

	// if(isFirstThread()) printf("tracker->numVoxels: %i \n", tracker->numVoxels);

	grid.sync();

	// numPoints = 30'000'000;

	auto t_project = Timer::start("project");

	{ // only project to single voxel
		uint32_t pointsPerThread = numPoints / totalThreadCount + 1;
		uint32_t pointsPerBlock = pointsPerThread * block.num_threads();
		uint32_t blockFirst = blockIdx.x * pointsPerBlock;

		for(int i = 0; i < pointsPerThread; i++){

			uint32_t pointIndex = blockFirst + i * block.num_threads() + threadIdx.x;

			if(pointIndex >= numPoints) break;

			Point point = input_points[pointIndex];

			float fx = fGridSize * (point.x) / boxSize.x;
			float fy = fGridSize * (point.y) / boxSize.y;
			float fz = fGridSize * (point.z) / boxSize.z;

			float sx = floor(fx - 0.5);
			float sy = floor(fy - 0.5);
			float sz = floor(fz - 0.5);

			vec3 pos = {fx, fy, fz};

			for(float ox = -1.0; ox <= 1.0; ox = ox + 1.0)
			for(float oy = -1.0; oy <= 1.0; oy = oy + 1.0)
			for(float oz = -1.0; oz <= 1.0; oz = oz + 1.0)
			// float ox = 0.0;
			// float oy = 0.0;
			// float oz = 0.0;
			{

				vec3 samplePos = vec3(
					floor(fx + ox) + 0.5,
					floor(fy + oy) + 0.5,
					floor(fz + oz) + 0.5
				);

				// float l = (pos - samplePos).length();
				float dx = (pos.x - samplePos.x);
				float dy = (pos.y - samplePos.y);
				float dz = (pos.z - samplePos.z);
				float ll = (dx * dx + dy * dy + dz * dz);
				float w = 0.0;

				if(ll < 1.0){
					// w = exp(-l * l * 3.5);
					// __expf is quite a bit faster than exp
					w = __expf(-ll * 3.5);
					w = clamp(w, 0.0, 1.0);
				}

				if(w > 0.0)
				{

					int W = clamp(100.0 * w, 1, 100);

					uint32_t ix = clamp(samplePos.x, 0.0f, fGridSize - 1.0f);
					uint32_t iy = clamp(samplePos.y, 0.0f, fGridSize - 1.0f);
					uint32_t iz = clamp(samplePos.z, 0.0f, fGridSize - 1.0f);

					uint32_t voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

					uint32_t R = W * ((point.color >>  0) & 0xff);
					uint32_t G = W * ((point.color >>  8) & 0xff);
					uint32_t B = W * ((point.color >> 16) & 0xff);

					if(flagGrid[voxelIndex] > 0){
						atomicAdd(&countingGrid[4 * voxelIndex + 0], R);
						atomicAdd(&countingGrid[4 * voxelIndex + 1], G);
						atomicAdd(&countingGrid[4 * voxelIndex + 2], B);
						atomicAdd(&countingGrid[4 * voxelIndex + 3], W);
					}
				}

			}
		}
	}

	grid.sync();

	t_project.stop();

	auto t_extract = Timer::start("extract");

	parallelIterator(0, gridSize * gridSize * gridSize, [points, lines, triangles, gridSize, boxSize, countingGrid](int index){
		int ix = index % gridSize;
		int iy = (index % (gridSize * gridSize)) / gridSize;
		int iz = index / (gridSize * gridSize);

		int voxelIndex = ix + gridSize * iy + gridSize * gridSize * iz;

		int W = countingGrid[4 * voxelIndex + 3];

		if(W > 0){
			float x = (float(ix) + 0.5) * boxSize.x / float(gridSize);
			float y = (float(iy) + 0.5) * boxSize.y / float(gridSize);
			float z = (float(iz) + 0.5) * boxSize.z / float(gridSize);
			float cubeSize = boxSize.x / float(gridSize);

			vec3 pos = {x, y, z};
			vec3 size = {cubeSize, cubeSize, cubeSize};
			uint32_t color = 0x000000ff;

			uint32_t R = float(countingGrid[4 * voxelIndex + 0]) / float(W);
			uint32_t G = float(countingGrid[4 * voxelIndex + 1]) / float(W);
			uint32_t B = float(countingGrid[4 * voxelIndex + 2]) / float(W);

			R = R & 0xff;
			G = G & 0xff;
			B = B & 0xff;

			color = R | (G << 8) | (B << 16);

			// drawBoundingBox(lines, pos, size, color);
			// if(voxelIndex % 10 == 0)
			// drawBox(triangles, pos, size, color);
			// if(abc == 0){
			// if(pos.x > 4.5 && pos.x < 4.6){
				drawPoint(points, pos, color);
			// }
			// drawPoint(points, pos, color);
			// atomicAdd(pointCount, 1);

		}

	});

	t_extract.stop();

	// grid.sync();

	// PRINT("pointCount: %i \n", pointCount[0]);


	grid.sync();

	// { // this clear runs at 640GB/s on an 930GB/s GPU
	// 	auto t_clear = Timer::start("clear");

	// 	for(int i = 0; i < 1000; i++){
	// 		uint32_t numVoxels = gridSize * gridSize * gridSize;
	// 		uint32_t pointsPerThread = numVoxels / totalThreadCount + 1;
	// 		uint32_t pointsPerBlock = pointsPerThread * block.num_threads();
	// 		uint32_t blockFirst = blockIdx.x * pointsPerBlock;

	// 		for(int i = 0; i < pointsPerThread; i++){

	// 			uint32_t voxelIndex = blockFirst + i * block.num_threads() + threadIdx.x;
	// 			if(voxelIndex < numVoxels){
	// 				flagGrid[voxelIndex] = i;
	// 			}

	// 		}

	// 		grid.sync();
	// 	}
	// 	t_clear.stop();
	// }

}

void main_voxelsampling(
	Points* points,
	Lines* lines, 
	Triangles* triangles, 
	Allocator& allocator, 
	Box3 box, 
	Point* input_points, 
	int numPoints
){

	// method_first_sample(points, lines, triangles, allocator, box, input_points, numPoints);
	// method_nearest(points, lines, triangles, allocator, box, input_points, numPoints);
	method_interpolate(points, lines, triangles, allocator, box, input_points, numPoints);

}
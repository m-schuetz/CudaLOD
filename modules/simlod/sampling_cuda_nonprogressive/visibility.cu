#define VERBOSE false

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "lib.h.cu"
#include "common.h"
#include "helper_math.h"

namespace cg = cooperative_groups;

float4 matMul(const Mat4& m, const float4& v){
	return make_float4(
		dot(m.rows[0], v), 
		dot(m.rows[1], v), 
		dot(m.rows[2], v), 
		dot(m.rows[3], v)
	);
}

int transposeIndex(int index){
	int a = index % 4;
	int b = index / 4;

	return 4 * a + b;
}

struct Plane{
	vec3 normal;
	float constant;
};

float t(int index, Mat4& transform){

	int a = index % 4;
	int b = index / 4;

	float* cols = reinterpret_cast<float*>(&transform);

	return cols[index];
}

float distanceToPoint(vec3 point, Plane plane){
	return plane.normal.dot(point) + plane.constant;
}

Plane createPlane(float x, float y, float z, float w){

	float nLength = vec3(x, y, z).length();

	Plane plane;
	plane.normal = vec3(x, y, z) / nLength;
	plane.constant = w / nLength;

	return plane;
}

bool intersectsFrustum(Mat4 worldViewProj, vec3 wgMin, vec3 wgMax){

	float* values = reinterpret_cast<float*>(&worldViewProj);

	float m_0  = values[transposeIndex( 0)];
	float m_1  = values[transposeIndex( 1)];
	float m_2  = values[transposeIndex( 2)];
	float m_3  = values[transposeIndex( 3)];
	float m_4  = values[transposeIndex( 4)];
	float m_5  = values[transposeIndex( 5)];
	float m_6  = values[transposeIndex( 6)];
	float m_7  = values[transposeIndex( 7)];
	float m_8  = values[transposeIndex( 8)];
	float m_9  = values[transposeIndex( 9)];
	float m_10 = values[transposeIndex(10)];
	float m_11 = values[transposeIndex(11)];
	float m_12 = values[transposeIndex(12)];
	float m_13 = values[transposeIndex(13)];
	float m_14 = values[transposeIndex(14)];
	float m_15 = values[transposeIndex(15)];

	Plane planes[6] = {
		createPlane(m_3 - m_0, m_7 - m_4, m_11 -  m_8, m_15 - m_12),
		createPlane(m_3 + m_0, m_7 + m_4, m_11 +  m_8, m_15 + m_12),
		createPlane(m_3 + m_1, m_7 + m_5, m_11 +  m_9, m_15 + m_13),
		createPlane(m_3 - m_1, m_7 - m_5, m_11 -  m_9, m_15 - m_13),
		createPlane(m_3 - m_2, m_7 - m_6, m_11 - m_10, m_15 - m_14),
		createPlane(m_3 + m_2, m_7 + m_6, m_11 + m_10, m_15 + m_14),
	};

	// Plane[] planes = frustumPlanes(worldViewProj);
	
	for(int i = 0; i < 6; i++){

		Plane plane = planes[i];

		vec3 vector;
		vector.x = plane.normal.x > 0.0 ? wgMax.x : wgMin.x;
		vector.y = plane.normal.y > 0.0 ? wgMax.y : wgMin.y;
		vector.z = plane.normal.z > 0.0 ? wgMax.z : wgMin.z;

		float d = distanceToPoint(vector, plane);

		if(d < 0){
			return false;
		}
	}

	return true;
}





struct Node{
	int pointOffset;
	int numPoints;
	Point* points;
	
	int numAdded;
	int level;
	int voxelIndex;
	vec3 min;
	vec3 max;
	float cubeSize;
	Node* children[8] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

	int numVoxels = 0;
	Point* voxels = nullptr;

	bool visible = true;

	bool isLeaf(){
		if(children[0] != nullptr) return false;
		if(children[1] != nullptr) return false;
		if(children[2] != nullptr) return false;
		if(children[3] != nullptr) return false;
		if(children[4] != nullptr) return false;
		if(children[5] != nullptr) return false;
		if(children[6] != nullptr) return false;
		if(children[7] != nullptr) return false;

		return true;
	}
};




extern "C" __global__
void kernel(
	uint32_t* buffer,
	Mat4 transform, 
	int2 imageSize,
	void** nnnodes,
	uint32_t* num_nodes,
	void** sssorted
){
	auto grid = cg::this_grid();

	grid.sync();

	int numNodes = *num_nodes;
	// numNodes = min(numNodes, 1000);

	Point* sorted = (Point*)*sssorted;
	Node* nodes = (Node*)*nnnodes;

	processRange(0, numNodes, [&](int nodeIndex){

		Node* node = &nodes[nodeIndex];

		vec3 center = (node->max + node->min) / 2.0;
		float4 pos = matMul(transform, {center.x, center.y, center.z, 1.0f});
		float distance = pos.w;

		bool visible = true;
		if(distance < 0.1){
			distance = 0.1;
		}
		if(node->cubeSize / distance < 0.1){
			visible = false;
		}

		if(!intersectsFrustum(transform, node->min, node->max)){
			visible = false;
		}

		if(node->level == 0){
			visible = true;
		}

		node->visible = visible;

		if(false)
		if(node->visible){
			drawBoundingBox(lines, 
				(node->max + node->min) / 2.0, 
				{node->cubeSize, node->cubeSize, node->cubeSize}, 
				0x0000ffff
			);
		}
	});

	grid.sync();

	int minLevel_points = 100;
	int maxLevel = 0;
	int maxVoxels = 0;
	int maxPoints = 0;
	int leafNodesPerLevel[10];
	memset(&leafNodesPerLevel, 0, 10 * 4);

	{
		// if(false)
		for(int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++){
			Node* node = &nodes[nodeIndex];

			maxVoxels = max(maxVoxels, node->numVoxels);
			maxPoints = max(maxPoints, node->numPoints);
			maxLevel  = max(maxLevel, node->level);
			if(node->numPoints > 0){
				minLevel_points = min(minLevel_points, node->level);
				leafNodesPerLevel[node->level]++;
			}

			// if(node != root) continue;
			if(!node->visible) continue;
			// if(node->level != 5) continue;

			uint32_t childMask = 0;
			for(int i = 0; i < 8; i++){
				Node* child = node->children[i];
				
				if(child && child->visible){
					childMask = childMask | (1 << i);
				}
			}


			// if(isFirstThread())
			// drawBoundingBox(lines, 
			// 	(node->max + node->min) / 2.0, 
			// 	{node->cubeSize, node->cubeSize, node->cubeSize}, 
			// 	0x000000ff
			// );

			grid.sync();
			// int verticesPerBox = 6 * 2 * 3;
			// int voxelOffset = voxels->count / verticesPerBox;
			// // int pointOffset = atomicAdd(&points->count, 0);

			// if(isFirstThread()){
			// 	voxels->count += node->numVoxels * 36;
			// 	// points->count += node->numVoxels;
			// }
			// grid.sync();

			// if(false)
			processRange(0, node->numVoxels, [&](int index){
				Point voxel = node->voxels[index];

				int ix = 2.0 * (voxel.x - node->min.x) / node->cubeSize;
				int iy = 2.0 * (voxel.y - node->min.y) / node->cubeSize;
				int iz = 2.0 * (voxel.z - node->min.z) / node->cubeSize;

				ix = ix > 0 ? 1 : 0;
				iy = iy > 0 ? 1 : 0;
				iz = iz > 0 ? 1 : 0;

				int childIndex = (ix << 2) | (iy << 1) | iz;
				
				bool childVisible = (childMask & (1 << childIndex)) != 0;

				if(childVisible) return;

				float4 pos = matMul(transform, {voxel.x, voxel.y, voxel.z, 1.0f});
				vec3 ndc = vec3(pos.x, pos.y, pos.z) / pos.w;

				// if(ndc.x < -1.0 || ndc.x > 1.0) return;
				// if(ndc.y < -1.0 || ndc.y > 1.0) return;
				// if(pos.w < 0.0) return;
				// if(pos.w > 1000.0) return;


				float voxelSize = node->cubeSize / 128.0;

				// Voxel voxelElement;
				// voxelElement.x = voxel.x;
				// voxelElement.y = voxel.y;
				// voxelElement.z = voxel.z;
				// voxelElement.color = voxel.color;
				// voxelElement.size = voxelSize;
				// voxels->voxels[voxelOffset + index] = voxelElement;

				// points->points[pointOffset + index] = voxel;

				uint32_t color = nodeIndex * 12345678;
				color = voxel.color;
				drawVoxel(voxels, {voxel.x, voxel.y, voxel.z}, voxelSize, voxel.color);
				// drawPoint(points, {voxel.x, voxel.y, voxel.z}, color);

			});

			grid.sync();

			// int pointOffset = atomicAdd(&points->count, 0);

			// if(isFirstThread()){
			// 	// voxels->count += node->numVoxels * 36;
			// 	points->count += node->numPoints;
			// }

			// grid.sync();

			processRange(0, node->numPoints, [&](int index){
				Point point = node->points[index];
				
				float4 pos = matMul(transform, {point.x, point.y, point.z, 1.0f});
				vec3 ndc = vec3(pos.x, pos.y, pos.z) / pos.w;

				// if(ndc.x < -1.0 || ndc.x > 1.0) return;
				// if(ndc.y < -1.0 || ndc.y > 1.0) return;
				// if(pos.w < 0.0) return;

				uint32_t color = nodeIndex * 12345678;
				color = point.color;
				drawPoint(points, {point.x, point.y, point.z}, color);
			});

			grid.sync();
		}
	}

	grid.sync();

	// if(isFirstThread()){

	// 	// printf("numNodes: %i \n", numNodes);

	// 	for(int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++){
	// 		Node* node = &nodes[nodeIndex];

	// 		if(node->numPoints > 0 && nodeIndex < 50){
	// 			printf("%i, ", node->numPoints);
	// 		}


	// 	}

	// 	// printf("\n");
	// }


	// for(int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++){
	// 	Node* node = &nodes[nodeIndex];

	// 	grid.sync();

	// 	if(node->level == 7){

	// 		if(isFirstThread()){
	// 			drawBoundingBox(lines, 
	// 				(node->max + node->min) / 2.0, 
	// 				{node->cubeSize, node->cubeSize, node->cubeSize}, 
	// 				0x00ff00ff
	// 			);

	// 			// printf("node->numPoints: %i \n", node->numPoints);

	// 		}

	// 		// processRange(0, node->numPoints, [&](int index){
	// 		// 	Point point = sorted[node->pointOffset + index];

	// 		// 	drawPoint(points, {point.x, point.y, point.z}, point.color);
	// 		// });
	// 	}
	// }



	grid.sync();

	// processRange(0, numNodes, [&](int nodeIndex){
	// 	Node* node = nodes[nodeIndex];



	// });

	if(false)
	if(isFirstThread())
	{

		for(int i = minLevel_points; i < 10; i++){
			int numNodes = leafNodesPerLevel[i];

			if(numNodes == 0) continue;

			// printf("%i: %i \n", i, numNodes);


		}

		// printf("minLevel_points: %i \n", minLevel_points);
		// printf("maxLevel: %i \n", maxLevel);
		// printf("maxVoxels: %i \n", maxVoxels);
		// printf("maxPoints: %i \n", maxPoints);
 		printf("maxLevel: %i, max(node.voxels): %i, max(node.points): %i \n", maxLevel, maxVoxels, maxPoints);

		int verticesPerBox = 6 * 2 * 3;
		int numVoxels = (voxels->count / 1000) / verticesPerBox;
		int numPoints = points->count / 1000;
		// printf("voxels: %i k, points: %i k \n", numVoxels, numPoints);
	}

}

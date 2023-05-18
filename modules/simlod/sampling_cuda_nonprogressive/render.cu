// Some code in this file, particularly frustum, ray and intersection tests, 
// is adapted from three.js. Three.js is licensed under the MIT license
// This file follows the three.js licensing
// License: MIT https://github.com/mrdoob/three.js/blob/dev/LICENSE


#define VERBOSE false

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "lib.h.cu"
#include "helper_math.h"
#include "common.h"

#include "methods_common.h.cu"

namespace cg = cooperative_groups;

constexpr int SPLAT_SIZE = 0;
constexpr float DEFAULT_DEPTH = 100000000000.0;
constexpr bool SHOW_BOUNDING_BOXES = false;
constexpr float minNodeSize = 64;
constexpr float DEPTH_THRESHOLD = 1.006f;
constexpr bool USE_BLENDING = true;

// https://colorbrewer2.org/#type=diverging&scheme=Spectral&n=11
// static constexpr uint32_t SPECTRAL[11] = {
// 	0x9e0142,
// 	0xd53e4f,
// 	0xf46d43,
// 	0xfdae61,
// 	0xfee08b,
// 	0xffffbf,
// 	0xe6f598,
// 	0xabdda4,
// 	0x66c2a5,
// 	0x3288bd,
// 	0x5e4fa2,
// };


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
	float* cols = reinterpret_cast<float*>(&transform);

	return cols[index];
}

float distanceToPoint(vec3 point, Plane plane){
	return plane.normal.dot(point) + plane.constant;
}

float distanceToPlane(vec3 origin, vec3 direction, Plane plane){

	float denominator = plane.normal.dot(direction);

	if(denominator == 0.0f){

		// line is coplanar, return origin
		if(distanceToPoint(origin, plane) == 0.0f){
			return 0.0f;
		}

		// Null is preferable to undefined since undefined means.... it is undefined
		return Infinity;
	}

	float t = -(origin.dot(plane.normal) + plane.constant) / denominator;

	if(t >= 0.0){
		return t;
	}else{
		return Infinity;
	}
}

Plane createPlane(float x, float y, float z, float w){

	float nLength = vec3(x, y, z).length();

	Plane plane;
	plane.normal = vec3(x, y, z) / nLength;
	plane.constant = w / nLength;

	return plane;
}

struct Frustum{
	Plane planes[6];

	static Frustum fromWorldViewProj(Mat4 worldViewProj){
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

		Frustum frustum;

		frustum.planes[0] = createPlane(m_3 - m_0, m_7 - m_4, m_11 -  m_8, m_15 - m_12);
		frustum.planes[1] = createPlane(m_3 + m_0, m_7 + m_4, m_11 +  m_8, m_15 + m_12);
		frustum.planes[2] = createPlane(m_3 + m_1, m_7 + m_5, m_11 +  m_9, m_15 + m_13);
		frustum.planes[3] = createPlane(m_3 - m_1, m_7 - m_5, m_11 -  m_9, m_15 - m_13);
		frustum.planes[4] = createPlane(m_3 - m_2, m_7 - m_6, m_11 - m_10, m_15 - m_14);
		frustum.planes[5] = createPlane(m_3 + m_2, m_7 + m_6, m_11 + m_10, m_15 + m_14);
		
		return frustum;
	}

	vec3 intersectRay(vec3 origin, vec3 direction){

		float closest = Infinity;

		for(int i = 0; i < 6; i++){

			Plane plane = planes[i];

			float d = distanceToPlane(origin, direction, plane);

			if(d > 0){
				closest = min(closest, d);
			}
		}

		vec3 intersection = {
			origin.x + direction.x * closest,
			origin.y + direction.y * closest,
			origin.z + direction.z * closest
		};

		return intersection;
	}

	bool contains(vec3 point){
		for(int i = 0; i < 6; i++){

			Plane plane = planes[i];

			// vec3 vector;
			// vector.x = plane.normal.x > 0.0 ? wgMax.x : wgMin.x;
			// vector.y = plane.normal.y > 0.0 ? wgMax.y : wgMin.y;
			// vector.z = plane.normal.z > 0.0 ? wgMax.z : wgMin.z;

			float d = distanceToPoint(point, plane);

			if(d < 0){
				return false;
			}
		}

		return true;
	}
};

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

struct RenderPassArgs{
	Mat4 transform;
	int2 imageSize;
	int numPoints;
	float LOD;
	int renderMode;
	bool showBoundingBox;
};

void drawLines(Point* vertices, int numLines, uint32_t* rt_depth, uint32_t* rt_color, RenderPassArgs& args){

	auto width = args.imageSize.x;
	auto height = args.imageSize.y;

	auto grid = cg::this_grid();
	grid.sync();

	// uint64_t t_start = 0;
	// if(isFirstThread()){
	// 	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));
	// }
	
	processRange(0, numLines, [&](int lineIndex){

		Point start = vertices[2 * lineIndex + 0];
		Point end = vertices[2 * lineIndex + 1];

		float4 ndc_start = matMul(args.transform, {start.x, start.y, start.z, 1.0f});
		ndc_start.x = ndc_start.x / ndc_start.w;
		ndc_start.y = ndc_start.y / ndc_start.w;

		float4 ndc_end   = matMul(args.transform, {end.x, end.y, end.z, 1.0f});
		ndc_end.x = ndc_end.x / ndc_end.w;
		ndc_end.y = ndc_end.y / ndc_end.w;

		vec3 screen_start = {
			(ndc_start.x * 0.5f + 0.5f) * width,
			(ndc_start.y * 0.5f + 0.5f) * height,
			1.0f
		};
		vec3 screen_end = {
			(ndc_end.x * 0.5f + 0.5f) * width,
			(ndc_end.y * 0.5f + 0.5f) * height,
			1.0f
		};

		// give up if one vertex is outside frustum, for now.
		// TODO: clip line to screen
		if(ndc_start.x < -1.0 || ndc_start.x > 1.0) return;
		if(ndc_start.y < -1.0 || ndc_start.y > 1.0) return;
		if(ndc_start.w < 0.0) return;
		if(ndc_end.x < -1.0 || ndc_end.x > 1.0) return;
		if(ndc_end.y < -1.0 || ndc_end.y > 1.0) return;
		if(ndc_end.w < 0.0) return;

		float steps = (screen_end - screen_start).length();
		// prevent long lines, to be safe
		steps = clamp(steps, 0.0f, 200.0f); 
		float stepSize = 1.0 / steps;

		// PRINT("%f \n", steps);

		float start_depth_linear = ndc_start.w;
		float end_depth_linear = ndc_end.w;

		for(float u = 0; u <= 1.0; u += stepSize){
			float ndc_x = (1.0 - u) * ndc_start.x + u * ndc_end.x;
			float ndc_y = (1.0 - u) * ndc_start.y + u * ndc_end.y;
			float depth = (1.0 - u) * start_depth_linear + u * end_depth_linear;

			if(ndc_x < -1.0 || ndc_x > 1.0) continue;
			if(ndc_y < -1.0 || ndc_y > 1.0) continue;

			int x = (ndc_x * 0.5 + 0.5) * width;
			int y = (ndc_y * 0.5 + 0.5) * height;

			x = clamp(x, 0, width - 1);
			y = clamp(y, 0, height - 1);

			int pixelID = x + width * y;

			uint32_t fb_depthi = rt_depth[pixelID];
			float fb_depth = *((float*)&fb_depthi);
			
			if(depth < fb_depth){
				rt_color[pixelID] = start.color;

				rt_depth[pixelID] = *((uint32_t*)&depth);
			}
		}
	});

	grid.sync();
}

void drawLines(Lines* lines, uint32_t* rt_depth, uint32_t* rt_color, RenderPassArgs& args){

	auto width = args.imageSize.x;
	auto height = args.imageSize.y;

	auto grid = cg::this_grid();
	grid.sync();

	// uint64_t t_start = 0;
	// if(isFirstThread()){
	// 	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));
	// }
 
	Frustum frustum = Frustum::fromWorldViewProj(args.transform);
	
	int numLines = lines->count / 2;
	processRange(0, numLines, [&](int lineIndex){

		Point start = lines->vertices[2 * lineIndex + 0];
		Point end = lines->vertices[2 * lineIndex + 1];

		vec3 dir = vec3{
			end.x - start.x,
			end.y - start.y,
			end.z - start.z
		}.normalize();

		if(!frustum.contains({start.x, start.y, start.z})){
			vec3 I = frustum.intersectRay({start.x, start.y, start.z}, dir);

			start.x = I.x;
			start.y = I.y;
			start.z = I.z;
		}

		if(!frustum.contains({end.x, end.y, end.z})){
			vec3 I = frustum.intersectRay({end.x, end.y, end.z}, dir * -1.0f);

			end.x = I.x;
			end.y = I.y;
			end.z = I.z;
		}
	


		float4 ndc_start = matMul(args.transform, {start.x, start.y, start.z, 1.0f});
		ndc_start.x = ndc_start.x / ndc_start.w;
		ndc_start.y = ndc_start.y / ndc_start.w;

		float4 ndc_end   = matMul(args.transform, {end.x, end.y, end.z, 1.0f});
		ndc_end.x = ndc_end.x / ndc_end.w;
		ndc_end.y = ndc_end.y / ndc_end.w;

		vec3 screen_start = {
			(ndc_start.x * 0.5f + 0.5f) * width,
			(ndc_start.y * 0.5f + 0.5f) * height,
			1.0f
		};
		vec3 screen_end = {
			(ndc_end.x * 0.5f + 0.5f) * width,
			(ndc_end.y * 0.5f + 0.5f) * height,
			1.0f
		};



		// give up if one vertex is outside frustum, for now.
		// TODO: clip line to screen
		// if(ndc_start.x < -1.0 || ndc_start.x > 1.0) return;
		// if(ndc_start.y < -1.0 || ndc_start.y > 1.0) return;
		// if(ndc_start.w < 0.0) return;
		// if(ndc_end.x < -1.0 || ndc_end.x > 1.0) return;
		// if(ndc_end.y < -1.0 || ndc_end.y > 1.0) return;
		// if(ndc_end.w < 0.0) return;

		float steps = (screen_end - screen_start).length();
		// prevent long lines, to be safe
		steps = clamp(steps, 0.0f, 200.0f); 
		float stepSize = 1.0 / steps;

		// PRINT("%f \n", steps);

		float start_depth_linear = ndc_start.w;
		float end_depth_linear = ndc_end.w;

		for(float u = 0; u <= 1.0; u += stepSize){
			float ndc_x = (1.0 - u) * ndc_start.x + u * ndc_end.x;
			float ndc_y = (1.0 - u) * ndc_start.y + u * ndc_end.y;
			float depth = (1.0 - u) * start_depth_linear + u * end_depth_linear;

			if(ndc_x < -1.0 || ndc_x > 1.0) continue;
			if(ndc_y < -1.0 || ndc_y > 1.0) continue;

			int x = (ndc_x * 0.5 + 0.5) * width;
			int y = (ndc_y * 0.5 + 0.5) * height;

			x = clamp(x, 0, width - 1);
			y = clamp(y, 0, height - 1);

			int pixelID = x + width * y;

			uint32_t fb_depthi = rt_depth[pixelID];
			float fb_depth = *((float*)&fb_depthi);
			
			if(depth < fb_depth){
				rt_color[pixelID] = start.color;

				rt_depth[pixelID] = *((uint32_t*)&depth);
			}
		}
	});

	grid.sync();
}

void drawPoints(Points* points, uint32_t* rt_depth, uint32_t* rt_color, RenderPassArgs& args){

	auto width = args.imageSize.x;
	auto height = args.imageSize.y;

	auto grid = cg::this_grid();
	grid.sync();

	processRange(0, points->count, [&](int pointIndex){

		Point point = points->points[pointIndex];

		float4 ndc = matMul(args.transform, {point.x, point.y, point.z, 1.0f});
		ndc.x = ndc.x / ndc.w;
		ndc.y = ndc.y / ndc.w;

		// clip to screen
		if(ndc.x < -1.0 || ndc.x > 1.0) return;
		if(ndc.y < -1.0 || ndc.y > 1.0) return;
		if(ndc.w < 0.0) return;

		float depth = ndc.w;

		int x = (ndc.x * 0.5f + 0.5f) * width;
		int y = (ndc.y * 0.5f + 0.5f) * height;

		x = clamp(x, 0, width - 1);
		y = clamp(y, 0, height - 1);

		int pixelID = x + width * y;

		uint32_t fb_depthi = rt_depth[pixelID];
		float fb_depth = *((float*)&fb_depthi);
		
		if(depth < fb_depth){
			rt_color[pixelID] = point.color;
			rt_depth[pixelID] = *((uint32_t*)&depth);
		}
		
	});

	grid.sync();
}

void drawBoundingBoxes(
	Allocator& allocator, 
	Node* nodes, 
	int numNodes, 
	uint32_t* depthBuffer, 
	uint32_t* colorBuffer, 
	RenderPassArgs& args
){
	
	// if(SHOW_BOUNDING_BOXES)
	{ // draw lines

		int linesPerBox = 12;
		int verticesPerBox = 2 * linesPerBox;
		
		int numLines = numNodes * linesPerBox;
		int numVertices = 2 * numLines;

		// Point* vertices = allocator.alloc<Point*>(sizeof(Point) * numVertices);
		Lines* lines = allocator.alloc<Lines*>(32 + sizeof(Point) * numVertices);
		lines->count = numVertices;
		Point* vertices = lines->vertices;

		processRange(0, numNodes, [&](int nodeIndex){
			
			Node* node = &nodes[nodeIndex];
			
			Point pMin = {node->min.x, node->min.y, node->min.z, 0x000000ff};
			Point pMax = {node->max.x, node->max.y, node->max.z, 0x000000ff};

			uint32_t color = 0x000000ff;

			if(node->numPoints > 0){
				color = 0x000000ff;
			}else if(node->numVoxels > 0){
				color = 0x0000ff00;
			}else{
				color = 0x0000ffff;
			}

			if(!node->visible)
			{
				color = 0x000000ff;
				pMin = {10000.0, 10000.0, 10000.0, 0};
				pMax = {10000.0, 10000.0, 10000.0, 0};
			}

			// BOTTOM
			vertices[nodeIndex * verticesPerBox +  0] = {pMin.x, pMin.y, pMin.z, color};
			vertices[nodeIndex * verticesPerBox +  1] = {pMax.x, pMin.y, pMin.z, color};
			vertices[nodeIndex * verticesPerBox +  2] = {pMax.x, pMin.y, pMin.z, color};
			vertices[nodeIndex * verticesPerBox +  3] = {pMax.x, pMax.y, pMin.z, color};
			vertices[nodeIndex * verticesPerBox +  4] = {pMax.x, pMax.y, pMin.z, color};
			vertices[nodeIndex * verticesPerBox +  5] = {pMin.x, pMax.y, pMin.z, color};
			vertices[nodeIndex * verticesPerBox +  6] = {pMin.x, pMax.y, pMin.z, color};
			vertices[nodeIndex * verticesPerBox +  7] = {pMin.x, pMin.y, pMin.z, color};

			// TOP
			vertices[nodeIndex * verticesPerBox +  8] = {pMin.x, pMin.y, pMax.z, color};
			vertices[nodeIndex * verticesPerBox +  9] = {pMax.x, pMin.y, pMax.z, color};
			vertices[nodeIndex * verticesPerBox + 10] = {pMax.x, pMin.y, pMax.z, color};
			vertices[nodeIndex * verticesPerBox + 11] = {pMax.x, pMax.y, pMax.z, color};
			vertices[nodeIndex * verticesPerBox + 12] = {pMax.x, pMax.y, pMax.z, color};
			vertices[nodeIndex * verticesPerBox + 13] = {pMin.x, pMax.y, pMax.z, color};
			vertices[nodeIndex * verticesPerBox + 14] = {pMin.x, pMax.y, pMax.z, color};
			vertices[nodeIndex * verticesPerBox + 15] = {pMin.x, pMin.y, pMax.z, color};

			// BOTTOM TO TOP
			vertices[nodeIndex * verticesPerBox + 16] = {pMin.x, pMin.y, pMin.z, color};
			vertices[nodeIndex * verticesPerBox + 17] = {pMin.x, pMin.y, pMax.z, color};
			vertices[nodeIndex * verticesPerBox + 18] = {pMax.x, pMin.y, pMin.z, color};
			vertices[nodeIndex * verticesPerBox + 19] = {pMax.x, pMin.y, pMax.z, color};
			vertices[nodeIndex * verticesPerBox + 20] = {pMax.x, pMax.y, pMin.z, color};
			vertices[nodeIndex * verticesPerBox + 21] = {pMax.x, pMax.y, pMax.z, color};
			vertices[nodeIndex * verticesPerBox + 22] = {pMin.x, pMax.y, pMin.z, color};
			vertices[nodeIndex * verticesPerBox + 23] = {pMin.x, pMax.y, pMax.z, color};
		});

		// drawLines(vertices, numLines, depthBuffer, colorBuffer, args);
		

		drawLines(lines, depthBuffer, colorBuffer, args);
	}
}

void renderBasic(
	Allocator& allocator, 
	RenderPassArgs& args,
	int nodeCount, 
	Node** visibleNodes,
	int width, int height,
	uint32_t* rt_resolved_depth,
	uint32_t* rt_resolved_color
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint64_t* rt_target = allocator.alloc<uint64_t*>(8 * width * height);

	processRange(0, width * height, [&](int pixelID){
		float depth = 100000000000.0;
		uint32_t depthi = *((uint32_t*)&depth);
		rt_resolved_depth[pixelID] = depthi;
		rt_resolved_color[pixelID] = 0;

		rt_target[pixelID] = uint64_t(depthi) << 32ull;
	});

	grid.sync();

	for(int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++){
		
		Node* node = visibleNodes[nodeIndex];

		if(!node->visible) continue;

		uint32_t childMask = 0;
		for(int i = 0; i < 8; i++){
			Node* child = node->children[i];
			
			if(child && child->visible){
				childMask = childMask | (1 << i);
			}
		}

		// VOXELS
		processRange(0, node->numVoxels, [&](int pointIndex){
			Point voxel = node->voxels[pointIndex];

			{// discard voxels where higher LOD childs are visible
				int ix = 2.0 * (voxel.x - node->min.x) / node->cubeSize;
				int iy = 2.0 * (voxel.y - node->min.y) / node->cubeSize;
				int iz = 2.0 * (voxel.z - node->min.z) / node->cubeSize;

				ix = ix > 0 ? 1 : 0;
				iy = iy > 0 ? 1 : 0;
				iz = iz > 0 ? 1 : 0;

				int childIndex = (ix << 2) | (iy << 1) | iz;
				
				bool childVisible = (childMask & (1 << childIndex)) != 0;

				if(childVisible) return;
			}

			float4 pos = matMul(args.transform, {voxel.x, voxel.y, voxel.z, 1.0f});
			pos.x = pos.x / pos.w;
			pos.y = pos.y / pos.w;

			if(pos.x < -1.0 || pos.x > 1.0) return;
			if(pos.y < -1.0 || pos.y > 1.0) return;
			if(pos.w < 0.0) return;

			float4 imgPos = {
				(pos.x * 0.5f + 0.5f) * width, 
				(pos.y * 0.5f + 0.5f) * height,
				pos.z, pos.w
			};

			int X = clamp(imgPos.x, 0.0, float(width) - 1.0);
			int Y = clamp(imgPos.y, 0.0, float(height) - 1.0);
			int pixelID = X + width * Y;

			float depth = pos.w;
			uint32_t idepth = *((uint32_t*)&depth);

			uint64_t encoded = (uint64_t(idepth) << 32ull) | voxel.color;

			atomicMin(&rt_target[pixelID], encoded);
		});

		// POINTS
		processRange(0, node->numPoints, [&](int pointIndex){
			Point point = node->points[pointIndex];
			float4 pos = matMul(args.transform, {point.x, point.y, point.z, 1.0f});
			pos.x = pos.x / pos.w;
			pos.y = pos.y / pos.w;

			if(pos.x < -1.0 || pos.x > 1.0) return;
			if(pos.y < -1.0 || pos.y > 1.0) return;
			if(pos.w < 0.0) return;

			float4 imgPos = {
				(pos.x * 0.5f + 0.5f) * width, 
				(pos.y * 0.5f + 0.5f) * height,
				pos.z, pos.w
			};

			int X = clamp(imgPos.x, 0.0, float(width) - 1.0);
			int Y = clamp(imgPos.y, 0.0, float(height) - 1.0);
			int pixelID = X + width * Y;

			float depth = pos.w;
			uint32_t idepth = *((uint32_t*)&depth);

			uint64_t encoded = (uint64_t(idepth) << 32ull) | point.color;

			atomicMin(&rt_target[pixelID], encoded);
		});
	}

	grid.sync();

	processRange(0, width * height, [&](int pixelID){
		uint64_t encoded = rt_target[pixelID];

		uint32_t idepth = encoded >> 32ull;
		uint32_t color = (encoded & 0xffffffffull);

		rt_resolved_depth[pixelID] = idepth;
		rt_resolved_color[pixelID] = color;
	});

	grid.sync();
}

void renderTest(
	Allocator& allocator, 
	RenderPassArgs& args,
	int nodeCount, 
	Node** visibleNodes,
	int width, int height,
	uint32_t* rt_resolved_depth,
	uint32_t* rt_resolved_color
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint64_t* rt_target     = allocator.alloc<uint64_t*>(8 * width * height);
	uint64_t* rt_target_2   = allocator.alloc<uint64_t*>(8 * width * height);
	uint32_t* rt_h_depth    = allocator.alloc<uint32_t*>(4 * width * height);
	uint32_t* rt_h_index    = allocator.alloc<uint32_t*>(4 * width * height);
	int32_t* rt_h_distance  = allocator.alloc<int32_t*>(4 * width * height);
	uint32_t* rt_v_depth    = allocator.alloc<uint32_t*>(4 * width * height);
	uint32_t* rt_v_index    = allocator.alloc<uint32_t*>(4 * width * height);
	int32_t* rt_v_distance  = allocator.alloc<int32_t*>(4 * width * height);

	processRange(0, width * height, [&](int pixelID){
		float depth = 100000000000.0;
		uint32_t depthi = *((uint32_t*)&depth);
		rt_resolved_depth[pixelID] = depthi;
		rt_resolved_color[pixelID] = 0;

		rt_target[pixelID] = uint64_t(depthi) << 32ull;
	});

	grid.sync();

	for(int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++){
		
		Node* node = visibleNodes[nodeIndex];

		if(!node->visible) continue;

		uint32_t childMask = 0;
		for(int i = 0; i < 8; i++){
			Node* child = node->children[i];
			
			if(child && child->visible){
				childMask = childMask | (1 << i);
			}
		}

		// VOXELS
		processRange(0, node->numVoxels, [&](int pointIndex){
			Point voxel = node->voxels[pointIndex];

			// if(false)
			{// discard voxels where higher LOD childs are visible
				int ix = 2.0 * (voxel.x - node->min.x) / node->cubeSize;
				int iy = 2.0 * (voxel.y - node->min.y) / node->cubeSize;
				int iz = 2.0 * (voxel.z - node->min.z) / node->cubeSize;

				ix = ix > 0 ? 1 : 0;
				iy = iy > 0 ? 1 : 0;
				iz = iz > 0 ? 1 : 0;

				int childIndex = (ix << 2) | (iy << 1) | iz;
				
				bool childVisible = (childMask & (1 << childIndex)) != 0;

				if(childVisible) return;
			}

			float4 pos = matMul(args.transform, {voxel.x, voxel.y, voxel.z, 1.0f});
			pos.x = pos.x / pos.w;
			pos.y = pos.y / pos.w;

			if(pos.x < -1.0 || pos.x > 1.0) return;
			if(pos.y < -1.0 || pos.y > 1.0) return;
			if(pos.w < 0.0) return;

			float4 imgPos = {
				(pos.x * 0.5f + 0.5f) * width, 
				(pos.y * 0.5f + 0.5f) * height,
				pos.z, pos.w
			};

			int X = clamp(imgPos.x, 0.0, float(width) - 1.0);
			int Y = clamp(imgPos.y, 0.0, float(height) - 1.0);
			int pixelID = X + width * Y;

			float depth = pos.w;
			uint64_t idepth = *((uint32_t*)&depth);

			uint64_t nodePointIndex = 
				(nodeIndex & 0b1111'1111'1111) |
				(pointIndex << 12);

			uint64_t encoded = (idepth << 32ull) | nodePointIndex;

			atomicMin(&rt_target[pixelID], encoded);
		});

		// POINTS
		processRange(0, node->numPoints, [&](int pointIndex){
			Point point = node->points[pointIndex];
			float4 pos = matMul(args.transform, {point.x, point.y, point.z, 1.0f});
			pos.x = pos.x / pos.w;
			pos.y = pos.y / pos.w;

			if(pos.x < -1.0 || pos.x > 1.0) return;
			if(pos.y < -1.0 || pos.y > 1.0) return;
			if(pos.w < 0.0) return;

			float4 imgPos = {
				(pos.x * 0.5f + 0.5f) * width, 
				(pos.y * 0.5f + 0.5f) * height,
				pos.z, pos.w
			};

			int X = clamp(imgPos.x, 0.0, float(width) - 1.0);
			int Y = clamp(imgPos.y, 0.0, float(height) - 1.0);
			int pixelID = X + width * Y;

			float depth = pos.w;
			uint32_t idepth = *((uint32_t*)&depth);

			uint64_t nodePointIndex = 
				(nodeIndex & 0b1111'1111'1111) |
				(pointIndex << 12);

			uint64_t encoded = (uint64_t(idepth) << 32ull) | nodePointIndex;

			atomicMin(&rt_target[pixelID], encoded);
		});
	}

	grid.sync();

	int window = 20;
	window = 1;
	window = 10;

	processRange(0, width * height, [&](int pixelID){
		// uint32_t count = 0;
		float newDepth = DEFAULT_DEPTH;
		uint32_t nodePointIndex = 0;
		int sampleDistance = -1000;

		for(int i = -window; i <= window; i++){
			uint64_t encoded = rt_target[pixelID + i];
			uint32_t idepth = encoded >> 32ull;
			float depth = *((float*)&idepth);

			float u = float(-i) / float(window);
			depth = depth + depth * u * u * 0.1;

			newDepth = min(newDepth, depth);

			if(depth != DEFAULT_DEPTH && depth == newDepth){
				nodePointIndex = encoded & 0xffffffffull;
				// nodePointIndex = 200;
				sampleDistance = -i;
			}
		}

		uint64_t idepth = *((uint32_t*)&newDepth);
		rt_h_depth[pixelID] = idepth;
		rt_h_index[pixelID] = nodePointIndex;
		rt_h_distance[pixelID] = sampleDistance;
	});

	grid.sync();

	processRange(0, width * height, [&](int pixelID){
		// uint32_t count = 0;
		float newDepth = DEFAULT_DEPTH;
		uint32_t nodePointIndex = 0;
		int sampleDistance = -1000;

		for(int i = -window; i <= window; i++){
			int samplePixelID = clamp(pixelID + i * width, 0, width * height);

			uint32_t idepth = rt_h_depth[samplePixelID];
			float depth = *((float*)&idepth);
			int32_t distance = rt_h_distance[samplePixelID];

			float u = float(distance) / float(window);
			float v = float(-i) / float(window);
			float dd = u * u + v * v;

			depth = depth + depth * v * v * 0.1;

			float candidateDepth = min(newDepth, depth);

			if(dd <= 1.05)
			if(depth != DEFAULT_DEPTH && depth == candidateDepth)
			{
				nodePointIndex = rt_h_index[samplePixelID];
				sampleDistance = i;

				// int r = int(255.0 * (0.5 * u + 0.5)) & 0xff;
				// int g = int(255.0 * (0.5 * v + 0.5)) & 0xff;
				// nodePointIndex = r | (g << 8);

				newDepth = candidateDepth;
			}
		}

		uint64_t idepth = *((uint32_t*)&newDepth);

		rt_v_depth[pixelID] = idepth;
		rt_v_index[pixelID] = nodePointIndex;
		rt_v_distance[pixelID] = sampleDistance;
	});

	grid.sync();

	processRange(0, width * height, [&](int pixelID){

		// int sampleDistance = rt_v_distance[pixelID];
		// int color = 10 * (float(sampleDistance) + float(window));
		// if(sampleDistance == -1000){
		// 	color = 0x00332211;
		// }

		uint32_t nodePointIndex = rt_v_index[pixelID];
		uint32_t nodeIndex = nodePointIndex & 0b1111'1111'1111;
		uint32_t pointIndex = nodePointIndex >> 12;

		nodeIndex = min(nodeIndex, nodeCount - 1);

		Node* node = visibleNodes[nodeIndex];
		Point point;
		if(node->numVoxels > 0){
			pointIndex = min(pointIndex, node->numVoxels - 1);
			point = node->voxels[pointIndex];
		}else{
			pointIndex = min(pointIndex, node->numPoints - 1);
			point = node->points[pointIndex];
			// point.color = 0x000000ff;
		}

		// uint32_t color = rt_v_index[pixelID];
		uint32_t color = point.color;
		// uint32_t color = pointIndex;
		// uint32_t color = 0x00ff00ff;

		
		float depth = ((float*)rt_v_depth)[pixelID];
		if(depth == DEFAULT_DEPTH){
			color = 0x00332211;
		}

		rt_resolved_depth[pixelID] = rt_v_depth[pixelID];
		rt_resolved_color[pixelID] = color;
		// rt_resolved_color[pixelID] = color;
	});

	// PRINT("%i \n", nodeCount);

	grid.sync();
}

void renderHQS(
	Allocator& allocator, 
	RenderPassArgs& args,
	int nodeCount, 
	Node** visibleNodes,
	int width, int height,
	uint32_t* rt_resolved_depth,
	uint32_t* rt_resolved_color
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint32_t* rt_depth = allocator.alloc<uint32_t*>( 4 * width * height);
	uint32_t* rt_accum = allocator.alloc<uint32_t*>(16 * width * height);

	processRange(0, width * height, [&](int pixelID){
		float depth = 100000000000.0;
		uint32_t depthi = *((uint32_t*)&depth);
		rt_depth[pixelID] = depthi;
		rt_accum[4 * pixelID + 0] = 0;
		rt_accum[4 * pixelID + 1] = 0;
		rt_accum[4 * pixelID + 2] = 0;
		rt_accum[4 * pixelID + 3] = 0;
	});

	grid.sync();

	{ // DEPTH
		__shared__ int sh_nodeIndex;
		uint32_t& nodeCounter = *allocator.alloc<uint32_t*>(4);
		nodeCounter = 0;
		grid.sync();

		// if(false)
		while(true){

			if(block.thread_rank() == 0){
				sh_nodeIndex = atomicAdd(&nodeCounter, 1);
			}
			block.sync();

			if(sh_nodeIndex >= nodeCount) break;

			Node* node = visibleNodes[sh_nodeIndex];

			// POINTS
			int numIterations_points = node->numPoints / block.num_threads() + 1;
			for(int it = 0; it < numIterations_points; it++){

				int pointIndex = it * block.num_threads() + block.thread_rank();

				if(pointIndex >= node->numPoints) break;

				Point point = node->points[pointIndex];
				float4 pos = matMul(args.transform, {point.x, point.y, point.z, 1.0f});
				pos.x = pos.x / pos.w;
				pos.y = pos.y / pos.w;

				if(pos.x < -1.0 || pos.x > 1.0) continue;
				if(pos.y < -1.0 || pos.y > 1.0) continue;
				if(pos.w < 0.0) continue;

				float4 imgPos = {
					(pos.x * 0.5f + 0.5f) * width, 
					(pos.y * 0.5f + 0.5f) * height,
					pos.z, pos.w
				};

				int X = clamp(imgPos.x, 0.0, float(width) - 1.0);
				int Y = clamp(imgPos.y, 0.0, float(height) - 1.0);
				int pixelID = X + width * Y;

				float depth = pos.w;
				uint32_t idepth = *((uint32_t*)&depth);

				atomicMin(&rt_depth[pixelID], idepth);
			}

			// VOXELS
			uint32_t childMask = 0;
			for(int i = 0; i < 8; i++){
				Node* child = node->children[i];
				
				if(child && child->visible){
					childMask = childMask | (1 << i);
				}
			}
			int numIterations_voxels = node->numVoxels / block.num_threads() + 1;
			for(int it = 0; it < numIterations_voxels; it++){

				int voxelIndex = it * block.num_threads() + block.thread_rank();

				if(voxelIndex >= node->numVoxels) break;

				Point voxel = node->voxels[voxelIndex];

				{// discard voxels where higher LOD childs are visible
					int ix = 2.0 * (voxel.x - node->min.x) / node->cubeSize;
					int iy = 2.0 * (voxel.y - node->min.y) / node->cubeSize;
					int iz = 2.0 * (voxel.z - node->min.z) / node->cubeSize;

					ix = ix > 0 ? 1 : 0;
					iy = iy > 0 ? 1 : 0;
					iz = iz > 0 ? 1 : 0;

					int childIndex = (ix << 2) | (iy << 1) | iz;
					
					bool childVisible = (childMask & (1 << childIndex)) != 0;

					if(childVisible) continue;
				}

				float4 pos = matMul(args.transform, {voxel.x, voxel.y, voxel.z, 1.0f});
				pos.x = pos.x / pos.w;
				pos.y = pos.y / pos.w;

				if(pos.x < -1.0 || pos.x > 1.0) continue;
				if(pos.y < -1.0 || pos.y > 1.0) continue;
				if(pos.w < 0.0) continue;

				float4 imgPos = {
					(pos.x * 0.5f + 0.5f) * width, 
					(pos.y * 0.5f + 0.5f) * height,
					pos.z, pos.w
				};

				int X = clamp(imgPos.x, 0.0, float(width) - 1.0);
				int Y = clamp(imgPos.y, 0.0, float(height) - 1.0);
				int pixelID = X + width * Y;

				float depth = pos.w;
				uint32_t idepth = *((uint32_t*)&depth);

				atomicMin(&rt_depth[pixelID], idepth);
			}
		}
	}

	grid.sync();

	{ // COLOR
		__shared__ int sh_nodeIndex;
		uint32_t& nodeCounter = *allocator.alloc<uint32_t*>(4);
		nodeCounter = 0;
		grid.sync();

		while(true){

			if(block.thread_rank() == 0){
				sh_nodeIndex = atomicAdd(&nodeCounter, 1);
			}
			block.sync();

			if(sh_nodeIndex >= nodeCount) break;

			Node* node = visibleNodes[sh_nodeIndex];

			// POINTS
			int numIterations_points = node->numPoints / block.num_threads() + 1;
			for(int it = 0; it < numIterations_points; it++){

				int pointIndex = it * block.num_threads() + block.thread_rank();

				if(pointIndex >= node->numPoints) break;

				Point point = node->points[pointIndex];
				float4 pos = matMul(args.transform, {point.x, point.y, point.z, 1.0f});
				pos.x = pos.x / pos.w;
				pos.y = pos.y / pos.w;

				if(pos.x < -1.0 || pos.x > 1.0) continue;
				if(pos.y < -1.0 || pos.y > 1.0) continue;
				if(pos.w < 0.0) continue;

				float4 imgPos = {
					(pos.x * 0.5f + 0.5f) * width, 
					(pos.y * 0.5f + 0.5f) * height,
					pos.z, pos.w
				};

				int X = clamp(imgPos.x, 0.0, float(width) - 1.0);
				int Y = clamp(imgPos.y, 0.0, float(height) - 1.0);
				int pixelID = X + width * Y;

				float depth = pos.w;

				uint32_t oldDepthi = rt_depth[pixelID];
				float oldDepth = *((float*)&oldDepthi);

				if(depth < oldDepth * DEPTH_THRESHOLD)
				{
					uint32_t R = (point.color >>  0) & 0xff;
					uint32_t G = (point.color >>  8) & 0xff;
					uint32_t B = (point.color >> 16) & 0xff;

					// R = 255;
					// G = 0;
					// B = 0;

					// R = clamp(R * 2u, 0u, 255u);
					// R = (node->level * 12346) % 200;
					// G = (node->level * 123467) % 200;
					// B = (node->level * 12346789) % 200;

					// R = (SPECTRAL[(node->level)] >> 0) & 0xff;
					// G = (SPECTRAL[(node->level)] >> 8) & 0xff;
					// B = (SPECTRAL[(node->level)] >> 16) & 0xff;

					atomicAdd(&rt_accum[4 * pixelID + 0], R);
					atomicAdd(&rt_accum[4 * pixelID + 1], G);
					atomicAdd(&rt_accum[4 * pixelID + 2], B);
					atomicAdd(&rt_accum[4 * pixelID + 3], 1);
				}
			}

			// VOXELS
			uint32_t childMask = 0;
			for(int i = 0; i < 8; i++){
				Node* child = node->children[i];
				
				if(child && child->visible){
					childMask = childMask | (1 << i);
				}
			}
			int numIterations_voxels = node->numVoxels / block.num_threads() + 1;
			for(int it = 0; it < numIterations_voxels; it++){
				int voxelIndex = it * block.num_threads() + block.thread_rank();

				if(voxelIndex >= node->numVoxels) break;

				Point voxel = node->voxels[voxelIndex];

				// if(false)
				{// discard voxels where higher LOD childs are visible
					int ix = 2.0 * (voxel.x - node->min.x) / node->cubeSize;
					int iy = 2.0 * (voxel.y - node->min.y) / node->cubeSize;
					int iz = 2.0 * (voxel.z - node->min.z) / node->cubeSize;

					ix = ix > 0 ? 1 : 0;
					iy = iy > 0 ? 1 : 0;
					iz = iz > 0 ? 1 : 0;

					int childIndex = (ix << 2) | (iy << 1) | iz;
					
					bool childVisible = (childMask & (1 << childIndex)) != 0;

					if(childVisible) continue;
				}

				float4 pos = matMul(args.transform, {voxel.x, voxel.y, voxel.z, 1.0f});
				pos.x = pos.x / pos.w;
				pos.y = pos.y / pos.w;
				
				if(pos.x < -1.0 || pos.x > 1.0) continue;
				if(pos.y < -1.0 || pos.y > 1.0) continue;
				if(pos.w < 0.0) continue;

				float4 imgPos = {
					(pos.x * 0.5f + 0.5f) * width, 
					(pos.y * 0.5f + 0.5f) * height,
					pos.z, pos.w
				};

				int X = clamp(imgPos.x, 0.0, float(width) - 1.0);
				int Y = clamp(imgPos.y, 0.0, float(height) - 1.0);
				int pixelID = X + width * Y;

				float depth = pos.w;

				uint32_t oldDepthi = rt_depth[pixelID];
				float oldDepth = *((float*)&oldDepthi);

				bool shouldAdd = false;

				if(!USE_BLENDING){
					shouldAdd = depth <= oldDepth;
				}else{
					shouldAdd = depth < oldDepth * DEPTH_THRESHOLD;
				}

				// if(depth <= oldDepth)
				// if((!USE_BLENDING && depth < oldDepth) || (depth < oldDepth * DEPTH_THRESHOLD))
				if(shouldAdd)
				{
					uint32_t R = (voxel.color >>  0) & 0xff;
					uint32_t G = (voxel.color >>  8) & 0xff;
					uint32_t B = (voxel.color >> 16) & 0xff;

					// R = 0;
					// G = 255;
					// B = 0;

					atomicAdd(&rt_accum[4 * pixelID + 0], R);
					atomicAdd(&rt_accum[4 * pixelID + 1], G);
					atomicAdd(&rt_accum[4 * pixelID + 2], B);
					atomicAdd(&rt_accum[4 * pixelID + 3], 1);
				}
			}
		}
	}

	grid.sync();

	{ // testing tile-based resolve
		uint32_t tileSize = 16;
		uint32_t numTiles_x = width / tileSize;
		uint32_t numTiles_y = height / tileSize;
		uint32_t numTiles = numTiles_x * numTiles_y;
		uint32_t tilesPerBlock = numTiles / grid.num_blocks();

		for(int i = 0; i < tilesPerBlock; i++){

			block.sync();

			int tileID = i * grid.num_blocks() + block.group_index().x;

			int tileX = tileID % numTiles_x;
			int tileY = tileID / numTiles_x;
			// int tileStart = tileSize * tileX + tileSize * tileSize * numTiles_x * tileY;
			int tileStart = tileX * tileSize + tileY * width * tileSize;

			int pixelID = tileStart 
				+ block.thread_rank() % tileSize
				+ (block.thread_rank() / tileSize) * width;
			
			float closestDepth = 1000000.0;
			for(int dx = 0; dx <= SPLAT_SIZE; dx++)
			for(int dy = 0; dy <= SPLAT_SIZE; dy++)
			{

				float fx = float(dx);
				float fy = float(dy);
				float ll = fx * fx + fy * fy;
				float nl = sqrt(ll) / SPLAT_SIZE;

				// if(nl >= 1.0) continue;
				
				int index = pixelID + dx + width * dy;
				index = max(index, 0);
				index = min(index, width * height);

				uint32_t depthi = rt_depth[index];
				float depth = *((float*)&depthi);

				// depth = depth + 0.00000 * nl * SPLAT_SIZE;

				closestDepth = min(closestDepth, depth);
			}

			uint64_t A = 0;
			uint64_t R = 0;
			uint64_t G = 0;
			uint64_t B = 0;
			for(int dx = 0; dx <= SPLAT_SIZE; dx++)
			for(int dy = 0; dy <= SPLAT_SIZE; dy++)
			{
				int index = pixelID + dx + width * dy;
				index = max(index, 0);
				index = min(index, width * height);

				float fx = float(dx);
				float fy = float(dy);
				float ll = fx * fx + fy * fy;
				float nl = sqrt(ll) / SPLAT_SIZE;

				// if(nl >= 1.0) continue;

				float w = __expf(-nl * nl * 50.5f);
				w = clamp(w, 0.001f, 1.0f);
				int W = 1000 * w;
				uint32_t depthi = rt_depth[index];
				float depth = *((float*)&depthi);

				float threshold = USE_BLENDING ? DEPTH_THRESHOLD : 1.0f;
				threshold = 1.0f;
				W = 1;

				// if(w > 0.0)
				// if(depth <= closestDepth * DEPTH_THRESHOLD)
				if(w > 0.0)
				if(depth <= closestDepth * threshold)
				{
					A += rt_accum[4 * index + 3] * W;
					R += rt_accum[4 * index + 0] * W;
					G += rt_accum[4 * index + 1] * W;
					B += rt_accum[4 * index + 2] * W;
				}

			}
			
			R = R / A;
			G = G / A;
			B = B / A;

			uint64_t depthi = rt_depth[pixelID];
			depthi = *(uint32_t*)(&closestDepth);
			uint32_t color = R | (G << 8) | (B << 16);
			// uint32_t color = 0x000000ff;

			if(A == 0){
				color = 0xffffffff;
				color = 0xff332211;
			}

			rt_resolved_depth[pixelID] = depthi;
			rt_resolved_color[pixelID] = color;

			block.sync();
		}
	}
}

void renderHQS(
	Allocator& allocator, 
	RenderPassArgs& args,
	int width, int height,
	uint32_t* rt_resolved_depth,
	uint32_t* rt_resolved_color,
	Point* points, uint32_t numPoints
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint32_t* rt_depth = allocator.alloc<uint32_t*>( 4 * width * height);
	uint32_t* rt_accum = allocator.alloc<uint32_t*>(16 * width * height);

	processRange(0, width * height, [&](int pixelID){
		float depth = 100000000000.0;
		uint32_t depthi = *((uint32_t*)&depth);
		rt_depth[pixelID] = depthi;
		rt_accum[4 * pixelID + 0] = 0;
		rt_accum[4 * pixelID + 1] = 0;
		rt_accum[4 * pixelID + 2] = 0;
		rt_accum[4 * pixelID + 3] = 0;
	});

	grid.sync();

	// DEPTH
	processRange(numPoints, [&](int pointIndex){
		Point point = points[pointIndex];

		float4 pos = matMul(args.transform, {point.x, point.y, point.z, 1.0f});
		pos.x = pos.x / pos.w;
		pos.y = pos.y / pos.w;

		if(pos.x < -1.0 || pos.x > 1.0) return;
		if(pos.y < -1.0 || pos.y > 1.0) return;
		if(pos.w < 0.0) return;

		float4 imgPos = {
			(pos.x * 0.5f + 0.5f) * width, 
			(pos.y * 0.5f + 0.5f) * height,
			pos.z, pos.w
		};

		int X = clamp(imgPos.x, 0.0, float(width) - 1.0);
		int Y = clamp(imgPos.y, 0.0, float(height) - 1.0);
		int pixelID = X + width * Y;

		float depth = pos.w;
		uint32_t idepth = *((uint32_t*)&depth);

		atomicMin(&rt_depth[pixelID], idepth);
	});
	
	grid.sync();

	// COLOR
	processRange(numPoints, [&](int pointIndex){
		Point point = points[pointIndex];
		
		float4 pos = matMul(args.transform, {point.x, point.y, point.z, 1.0f});
		pos.x = pos.x / pos.w;
		pos.y = pos.y / pos.w;

		if(pos.x < -1.0 || pos.x > 1.0) return;
		if(pos.y < -1.0 || pos.y > 1.0) return;
		if(pos.w < 0.0) return;

		float4 imgPos = {
			(pos.x * 0.5f + 0.5f) * width, 
			(pos.y * 0.5f + 0.5f) * height,
			pos.z, pos.w
		};

		int X = clamp(imgPos.x, 0.0, float(width) - 1.0);
		int Y = clamp(imgPos.y, 0.0, float(height) - 1.0);
		int pixelID = X + width * Y;

		float depth = pos.w;

		uint32_t oldDepthi = rt_depth[pixelID];
		float oldDepth = *((float*)&oldDepthi);

		if(depth < oldDepth * DEPTH_THRESHOLD)
		// if(depth < oldDepth + 2.1)
		{
			uint32_t R = (point.color >>  0) & 0xff;
			uint32_t G = (point.color >>  8) & 0xff;
			uint32_t B = (point.color >> 16) & 0xff;

			atomicAdd(&rt_accum[4 * pixelID + 0], R);
			atomicAdd(&rt_accum[4 * pixelID + 1], G);
			atomicAdd(&rt_accum[4 * pixelID + 2], B);
			atomicAdd(&rt_accum[4 * pixelID + 3], 1);

			// if(old > 
		}
	});

	grid.sync();

	{ // testing tile-based resolve
		uint32_t tileSize = 16;
		uint32_t numTiles_x = width / tileSize;
		uint32_t numTiles_y = height / tileSize;
		uint32_t numTiles = numTiles_x * numTiles_y;
		uint32_t tilesPerBlock = numTiles / grid.num_blocks();

		for(int i = 0; i < tilesPerBlock; i++){

			block.sync();

			int tileID = i * grid.num_blocks() + block.group_index().x;

			int tileX = tileID % numTiles_x;
			int tileY = tileID / numTiles_x;
			// int tileStart = tileSize * tileX + tileSize * tileSize * numTiles_x * tileY;
			int tileStart = tileX * tileSize + tileY * width * tileSize;

			int pixelID = tileStart 
				+ block.thread_rank() % tileSize
				+ (block.thread_rank() / tileSize) * width;
			
			float closestDepth = 1000000.0;
			for(int dx = 0; dx <= SPLAT_SIZE; dx++)
			for(int dy = 0; dy <= SPLAT_SIZE; dy++)
			{

				float fx = float(dx);
				float fy = float(dy);
				float ll = fx * fx + fy * fy;
				float nl = sqrt(ll) / SPLAT_SIZE;

				// if(nl >= 1.0) continue;
				
				int index = pixelID + dx + width * dy;
				index = max(index, 0);
				index = min(index, width * height);

				uint32_t depthi = rt_depth[index];
				float depth = *((float*)&depthi);

				// depth = depth + 0.00000 * nl * SPLAT_SIZE;

				closestDepth = min(closestDepth, depth);
			}

			uint64_t A = 0;
			uint64_t R = 0;
			uint64_t G = 0;
			uint64_t B = 0;
			for(int dx = 0; dx <= SPLAT_SIZE; dx++)
			for(int dy = 0; dy <= SPLAT_SIZE; dy++)
			{
				int index = pixelID + dx + width * dy;
				index = max(index, 0);
				index = min(index, width * height);

				float fx = float(dx);
				float fy = float(dy);
				float ll = fx * fx + fy * fy;
				float nl = sqrt(ll) / SPLAT_SIZE;

				// if(nl >= 1.0) continue;

				float w = __expf(-nl * nl * 50.5f);
				w = clamp(w, 0.001f, 1.0f);
				int W = 1000 * w;
				uint32_t depthi = rt_depth[index];
				float depth = *((float*)&depthi);

				W = 1;

				if(w > 0.0)
				if(depth <= closestDepth * DEPTH_THRESHOLD)
				{
					A += rt_accum[4 * index + 3] * W;
					R += rt_accum[4 * index + 0] * W;
					G += rt_accum[4 * index + 1] * W;
					B += rt_accum[4 * index + 2] * W;
				}

			}
			
			R = R / A;
			G = G / A;
			B = B / A;

			uint64_t depthi = rt_depth[pixelID];
			depthi = *(uint32_t*)(&closestDepth);
			uint32_t color = R | (G << 8) | (B << 16);
			// uint32_t color = 0x000000ff;

			if(A == 0){
				color = 0xffffffff;
				color = 0xff332211;
			}

			rt_resolved_depth[pixelID] = depthi;
			rt_resolved_color[pixelID] = color;

			block.sync();
		}
	}


}

extern "C" __global__
void kernel(
	RenderPassArgs args,
	cudaSurfaceObject_t surface,
	uint32_t* buffer,
	void** nnnodes,
	uint32_t* num_nodes,
	void** sssorted,
	void** _points,
	void** _lines,
	Point* _input_points
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int numNodes = *num_nodes;
	Node* nodes = (Node*)*nnnodes;
	Lines* lines = (Lines*)*_lines;
	Points* points = (Points*)*_points;

	lines->count = 0;

	Allocator allocator(buffer, 0);

	uint32_t& nodeCount = *allocator.alloc<uint32_t*>(4);
	uint32_t& pointCount = *allocator.alloc<uint32_t*>(4);
	uint32_t& voxelCount = *allocator.alloc<uint32_t*>(4);

	if(isFirstThread()){
		pointCount = 0;
		voxelCount = 0;
		nodeCount = 0;
	}

	auto width = args.imageSize.x;
	auto height = args.imageSize.y;

	// allocate framebuffers
	uint32_t* rt_resolved_depth = allocator.alloc<uint32_t*>( 4 * width * height);
	uint32_t* rt_resolved_color = allocator.alloc<uint32_t*>( 4 * width * height);

	grid.sync();

	uint32_t pointsPerBlock = args.numPoints / grid.num_blocks();
	uint32_t pointsPerThread = pointsPerBlock / block.size();
	uint32_t pointsPerIteration = block.num_threads();

	uint32_t offset_block = block.group_index().x * pointsPerBlock;

	Node** visibleNodes = allocator.alloc<Node**>(sizeof(Node*) * numNodes);

	grid.sync();

	auto min8 = [](float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7){

		float m0 = min(f0, f1);
		float m1 = min(f2, f3);
		float m2 = min(f4, f5);
		float m3 = min(f6, f7);

		float n0 = min(m0, m1);
		float n1 = min(m2, m3);

		return min(n0, n1);
	};

	auto max8 = [](float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7){

		float m0 = max(f0, f1);
		float m1 = max(f2, f3);
		float m2 = max(f4, f5);
		float m3 = max(f6, f7);

		float n0 = max(m0, m1);
		float n1 = max(m2, m3);

		return max(n0, n1);
	};

	float fwidth = args.imageSize.x;
	float fheight = args.imageSize.y;

	// COMPUTE LIST OF VISIBLE NODES
	processRange(0, numNodes, [&](int nodeIndex){

		Node* node = &nodes[nodeIndex];

		vec3 center = (node->max + node->min) / 2.0;
		float4 pos = matMul(args.transform, {center.x, center.y, center.z, 1.0f});
		float distance = pos.w;

		bool visible = true;
		if(distance < 0.1){
			distance = 0.1;
		}

		float4 p000 = {node->min.x, node->min.y, node->min.z, 1.0f};
		float4 p001 = {node->min.x, node->min.y, node->max.z, 1.0f};
		float4 p010 = {node->min.x, node->max.y, node->min.z, 1.0f};
		float4 p011 = {node->min.x, node->max.y, node->max.z, 1.0f};
		float4 p100 = {node->max.x, node->min.y, node->min.z, 1.0f};
		float4 p101 = {node->max.x, node->min.y, node->max.z, 1.0f};
		float4 p110 = {node->max.x, node->max.y, node->min.z, 1.0f};
		float4 p111 = {node->max.x, node->max.y, node->max.z, 1.0f};

		float4 ndc000 = matMul(args.transform, p000);
		float4 ndc001 = matMul(args.transform, p001);
		float4 ndc010 = matMul(args.transform, p010);
		float4 ndc011 = matMul(args.transform, p011);
		float4 ndc100 = matMul(args.transform, p100);
		float4 ndc101 = matMul(args.transform, p101);
		float4 ndc110 = matMul(args.transform, p110);
		float4 ndc111 = matMul(args.transform, p111);

		float4 s000 = ((ndc000 / ndc000.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s001 = ((ndc001 / ndc001.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s010 = ((ndc010 / ndc010.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s011 = ((ndc011 / ndc011.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s100 = ((ndc100 / ndc100.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s101 = ((ndc101 / ndc101.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s110 = ((ndc110 / ndc110.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};
		float4 s111 = ((ndc111 / ndc111.w) * 0.5f + 0.5f) * float4{fwidth, fheight, 1.0f, 1.0f};

		float smin_x = min8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
		float smin_y = min8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

		float smax_x = max8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
		float smax_y = max8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

		float dx = smax_x - smin_x;
		float dy = smax_y - smin_y;

		// if(nodeIndex == 0){
		// 	printf("%f, %f \n" , dx, dy);
		// }

		// float lod_factor = 1.0f - 0.97f * args.LOD;
		// lod_factor = 2.0 * 0.062;
		// lod_factor = 1.0;
		// lod_factor = s000.w;
		// PRINT("lod_factor: %f \n", lod_factor);

		// if(node->cubeSize / distance < lod_factor){
		// 	visible = false;
		// }

		// if(dx < 64 || dy < 64)
		// if(dx < 128 || dy < 128)
		if(dx < minNodeSize || dy < minNodeSize)
		{
			visible = false;
		}

		if(!intersectsFrustum(args.transform, node->min, node->max)){
			visible = false;
		}

		if(node->level == 0){
			visible = true;
		}
		
		// if(node->isLeaf())
		// {
		// 	vec3 pos = (node->min + node->max) * 0.5f;
		// 	vec3 size = node->max - node->min;
		// 	drawBoundingBox(lines, pos, size, 0x000000ff);

		// 	if(node->numPoints != node->numAdded){
		// 		printf("subgrid[%i].voxelIndex[%i]: numPoints != numAdded -> %i != %i \n", 
		// 			node->dbg, node->voxelIndex, node->numPoints, node->numAdded);
		// 	}
		// }

		int sumChildPoints = 0;
		int sumChildVoxels = 0;
		for(int childIndex = 0; childIndex < 8; childIndex++){
			auto child = node->children[childIndex];

			if(child == nullptr) continue;

			sumChildPoints += child->numPoints;
			sumChildVoxels += child->numVoxels;
		}

		node->visible = visible;

		if(node->visible){
			uint32_t index = atomicAdd(&nodeCount, 1);
			visibleNodes[index] = node;

			atomicAdd(&pointCount, node->numPoints);
			atomicAdd(&voxelCount, node->numVoxels);
		}
	});

	grid.sync();

	// renderBasic(allocator, args, nodeCount, visibleNodes, width, height, rt_resolved_depth, rt_resolved_color);
	// renderTest(allocator, args, nodeCount, visibleNodes, width, height, rt_resolved_depth, rt_resolved_color);
	// renderHQS(allocator, args, nodeCount, visibleNodes, width, height, rt_resolved_depth, rt_resolved_color);

	if(args.renderMode == 0){
		renderHQS(allocator, args, nodeCount, visibleNodes, width, height, rt_resolved_depth, rt_resolved_color);
	}else if(args.renderMode == 1){
		renderHQS(allocator, args, width, height, 
			rt_resolved_depth, rt_resolved_color,
			_input_points, uint32_t(args.numPoints)
		);
	}

	grid.sync();

	if(args.showBoundingBox){
		drawBoundingBoxes(allocator, nodes, numNodes, rt_resolved_depth, rt_resolved_color, args);
	}

	grid.sync();
	drawLines(lines, rt_resolved_depth, rt_resolved_color, args);
	grid.sync();
	drawPoints(points, rt_resolved_depth, rt_resolved_color, args);
	grid.sync();
	uint32_t* rt_edl = allocator.alloc<uint32_t*>(4 * width * height);



	// // EDL
	// processRange(0, width * height, [&](int pixelID){
	// 	float pixelDepth = *((float*)(&rt_resolved_depth[pixelID]));
	// 	uint32_t pixelColor = rt_resolved_color[pixelID];
		
	// 	float PI = 3.1415;
	// 	float numSamples = 5;
	// 	float stepsize = 2.0 * PI / numSamples;
	// 	float r = 3.0;
	// 	float edlStrength = 0.4;

	// 	float sum = 0.0;
	// 	for(float u = 0.0; u < 2.0 * PI; u += stepsize)
	// 	{
	// 		int dx = r * sin(u);
	// 		int dy = r * cos(u);

	// 		int index = pixelID + dx + width * dy;
	// 		index = max(index, 0);
	// 		index = min(index, width * height);

	// 		float neighbourDepth = ((float*)(rt_resolved_depth))[index];
			
	// 		sum = sum + max(log2(pixelDepth) - log2(neighbourDepth), 0.0);
	// 	}

	// 	float response = sum / numSamples;
	// 	float shade = __expf(-response * 300.0 * edlStrength);

	// 	uint32_t R = shade * ((pixelColor >>  0) & 0xff);
	// 	uint32_t G = shade * ((pixelColor >>  8) & 0xff);
	// 	uint32_t B = shade * ((pixelColor >> 16) & 0xff);
	// 	uint32_t color = R | (G << 8) | (B << 16) | (255 << 24);

	// 	rt_edl[pixelID] = color;
	// });

	if(false)
	{ // tile-based EDL
		uint32_t tileSize = 16;
		uint32_t numTiles_x = width / tileSize;
		uint32_t numTiles_y = height / tileSize;
		uint32_t numTiles = numTiles_x * numTiles_y;
		uint32_t tilesPerBlock = numTiles / grid.num_blocks();

		for(int i = 0; i < tilesPerBlock; i++){

			block.sync();

			int tileID = i * grid.num_blocks() + block.group_index().x;
			int tileX = tileID % numTiles_x;
			int tileY = tileID / numTiles_x;
			int tileStart = tileX * tileSize + tileY * width * tileSize;

			int pixelID = tileStart + block.thread_rank() % tileSize + (block.thread_rank() / tileSize) * width;

			float pixelDepth = *((float*)(&rt_resolved_depth[pixelID]));
			uint32_t pixelColor = rt_resolved_color[pixelID];

			float PI = 3.1415;
			float numSamples = 5;
			float stepsize = 2.0 * PI / numSamples;
			float r = 3.0;
			float edlStrength = 0.2;

			float sum = 0.0;
			for(float u = 0.0; u < 2.0 * PI; u += stepsize)
			{
				int dx = r * sin(u);
				int dy = r * cos(u);

				int index = pixelID + dx + width * dy;
				index = max(index, 0);
				index = min(index, width * height);

				float neighbourDepth = ((float*)(rt_resolved_depth))[index];
				
				sum = sum + max(log2(pixelDepth) - log2(neighbourDepth), 0.0);
			}
			
			float response = sum / numSamples;
			float shade = __expf(-response * 300.0 * edlStrength);
			shade = 1.0;

			uint32_t R = shade * ((pixelColor >>  0) & 0xff);
			uint32_t G = shade * ((pixelColor >>  8) & 0xff);
			uint32_t B = shade * ((pixelColor >> 16) & 0xff);
			uint32_t color = R | (G << 8) | (B << 16) | (255u << 24);

			rt_edl[pixelID] = color;

			block.sync();
		}
		grid.sync();
		rt_resolved_color = rt_edl;
	}


	if(false)
	{ // render zoomed inset

		int source_x = 800;
		int source_y = 800;
		int source_w = 100;
		int source_h = 100;
		int target_x = 1700;
		int target_y = 10;
		int target_w = 800;
		int target_h = 800;

		processRange(0, target_w * target_h, [&](int index){
			int x = index % target_w;
			int y = index / target_w;

			float u = float(x) / float(target_w);
			float v = float(y) / float(target_h);

			int sx = float(source_w) * u + source_x;
			int sy = float(source_h) * v + source_y;
			int tx = x + target_x;
			int ty = y + target_y;

			int sourcePixelID = sx + width * sy;
			int targetPixelID = tx + width * ty;

			rt_resolved_depth[targetPixelID] = 0;
			rt_resolved_color[targetPixelID] = rt_resolved_color[sourcePixelID];
		});
	}

	grid.sync();

	// { // transfer to opengl texture

	// 	processRange(0, width * height, [&](int pixelID){

	// 		uint32_t depthi = rt_resolved_depth[pixelID];
	// 		uint32_t color = rt_resolved_color[pixelID];

	// 		int x = pixelID % width;
	// 		int y = pixelID / width;
	// 		x = clamp(x, 0, width - 1);
	// 		y = clamp(y, 0, height - 1);
	// 		surf2Dwrite(color, surface, x * 4, y);

	// 	});

	// }

	{ // tile-based transfer to opengl texture
		uint32_t tileSize = 16;
		uint32_t numTiles_x = width / tileSize;
		uint32_t numTiles_y = height / tileSize;
		uint32_t numTiles = numTiles_x * numTiles_y;
		uint32_t tilesPerBlock = numTiles / grid.num_blocks();

		for(int i = 0; i < tilesPerBlock; i++){

			block.sync();

			int tileID = i * grid.num_blocks() + block.group_index().x;

			int tileX = tileID % numTiles_x;
			int tileY = tileID / numTiles_x;
			// int tileStart = tileSize * tileX + tileSize * tileSize * numTiles_x * tileY;
			int tileStart = tileX * tileSize + tileY * width * tileSize;

			int pixelID = tileStart 
				+ block.thread_rank() % tileSize
				+ (block.thread_rank() / tileSize) * width;
			
			// uint32_t depthi = rt_resolved_depth[pixelID];
			uint32_t color = rt_resolved_color[pixelID];

			int x = pixelID % width;
			int y = pixelID / width;
			x = clamp(x, 0, width - 1);
			y = clamp(y, 0, height - 1);
			surf2Dwrite(color, surface, x * 4, y);

			block.sync();
		}
	}


	// // if(false)
	// { // transfer to opengl texture, WITH EDL
	// 	uint32_t tileSize = 16;
	// 	uint32_t numTiles_x = width / tileSize;
	// 	uint32_t numTiles_y = height / tileSize;
	// 	uint32_t numTiles = numTiles_x * numTiles_y;
	// 	uint32_t tilesPerBlock = numTiles / grid.num_blocks();

	// 	for(int i = 0; i < tilesPerBlock; i++){

	// 		block.sync();

	// 		int tileID = i * grid.num_blocks() + block.group_index().x;

	// 		int tileX = tileID % numTiles_x;
	// 		int tileY = tileID / numTiles_x;
	// 		int tileStart = tileSize * tileX + tileSize * tileSize * numTiles_x * tileY;

	// 		int pixelID = tileStart + block.thread_rank() % tileSize + (block.thread_rank() / tileSize) * width;

	// 		float pixelDepth = ((float*)(rt_resolved_depth))[pixelID];
	// 		uint32_t pixelColor = rt_resolved_color[pixelID];

	// 		float PI = 3.1415;
	// 		float numSamples = 5;
	// 		float stepsize = 2.0 * PI / numSamples;
	// 		float r = 2.0;

	// 		float sum = 0.0;
	// 		for(float u = 0.0; u < 2.0 * PI; u += stepsize)
	// 		{
	// 			int dx = r * sin(u);
	// 			int dy = r * cos(u);

	// 			int index = pixelID + dx + width * dy;
	// 			index = max(index, 0);
	// 			index = min(index, width * height);

	// 			float neighbourDepth = ((float*)(rt_resolved_depth))[index];
				
	// 			sum = sum + max(log2(pixelDepth) - log2(neighbourDepth), 0.0);
	// 		}
			
	// 		float edlStrength = 0.2;
	// 		float response = sum / numSamples;
	// 		float shade = __expf(-response * 300.0 * edlStrength);

	// 		uint32_t R = shade * ((pixelColor >>  0) & 0xff);
	// 		uint32_t G = shade * ((pixelColor >>  8) & 0xff);
	// 		uint32_t B = shade * ((pixelColor >> 16) & 0xff);
	// 		uint32_t color = R | (G << 8) | (B << 16) | (255 << 24);

	// 		// uint32_t depthi = rt_resolved_depth[pixelID];
	// 		// uint32_t color = rt_resolved_color[pixelID];

	// 		int x = pixelID % width;
	// 		int y = pixelID / width;
	// 		x = clamp(x, 0, width - 1);
	// 		y = clamp(y, 0, height - 1);
	// 		surf2Dwrite(color, surface, x * 4, y);

	// 		block.sync();
	// 	}
	// }


	// if(false)
	if(isFirstThread())
	{
		// printf("pointCount: %i, voxelCount: %i \n", pointCount, voxelCount);

		printf("\r#nodes: ");
		printNumber(nodeCount, 4);
		printf(", #points: ");
		printNumber(pointCount, 10);
		printf(", #voxels: ");
		printNumber(voxelCount, 10);
		// printf("\n");
	}

}


#pragma once

struct Mat4 {
	float4 rows[4];
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

enum SamplingStrategy{
	FIRST_COME            = 0,
	RANDOM                = 1,
	AVERAGE_SINGLECELL    = 2,
	WEIGHTED_NEIGHBORHOOD = 3
};

struct State{
	PointCloudMetadata metadata;
	Mat4 transform;
	int2 imageSize;
	SamplingStrategy strategy;
	float LOD;
};

constexpr int HISTOGRAM_NUM_BINS = 100;
struct Results{
	SamplingStrategy strategy;
	uint64_t points;
	uint64_t voxels;
	uint64_t nodes;
	uint64_t innerNodes;
	uint64_t leafNodes;
	uint64_t minPoints;
	uint64_t maxPoints;
	uint64_t avgPoints;
	uint64_t minVoxels;
	uint64_t maxVoxels;
	uint64_t avgVoxels;

	uint64_t histogram_maxval;
	uint64_t histogram_points[HISTOGRAM_NUM_BINS];
	uint64_t histogram_voxels[HISTOGRAM_NUM_BINS];

	uint64_t pointsPerLevel[50];
	uint64_t voxelsPerLevel[50];
	uint64_t innerNodesAtLevel[50];
	uint64_t leafNodesAtLevel[50];

	uint64_t allocatedMemory_splitting;
	uint64_t allocatedMemory_voxelization;
};
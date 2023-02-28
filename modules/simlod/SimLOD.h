
#pragma once

#include <string>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "unsuck.hpp"

#include "Box.h"
#include "Debug.h"
#include "Camera.h"
#include "LasLoader.h"
#include "Frustum.h"
#include "Renderer.h"

#include "simlod/voxel_sampling_gentree/voxel_sampling_gentree.h"
#include "simlod/sampling_cuda/sampling_cuda.h"
#include "simlod/sampling_cuda_nonprogressive/sampling_cuda_nonprogressive.h"

using namespace std;
using namespace std::chrono_literals;

namespace simlod{

// from: https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
// license: https://creativecommons.org/licenses/by-nc-sa/3.0/
// method to seperate bits from a given integer 3 positions apart
inline uint64_t splitBy3(unsigned int a){
	uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
	x = (x | x << 32) & 0x1f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
	x = (x | x << 16) & 0x1f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
	x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
	x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

inline uint64_t mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z){
	uint64_t answer = 0;
	answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
	return answer;
}


struct SimLOD{

	string path = "";
	Renderer* renderer = nullptr;

	shared_ptr<LasFile> lasfile = nullptr;

	// GLuint ssPoints = -1;
	uint64_t numPoints = 0;
	
	Shader* shader = nullptr;

	SimLOD() {

	}

	SimLOD(string path, Renderer* renderer) {
		this->path = path;
		this->renderer = renderer;

		// glCreateBuffers(1, &ssPoints);

		this->lasfile = loadLas(path);

		this->numPoints = lasfile->points.size();
		// this->numPoints = min(this->numPoints, 250'000'000ull);
		uint64_t bufferSize = 16ull * this->numPoints;
		// glNamedBufferData(ssPoints, bufferSize, &lasfile->points[0], GL_DYNAMIC_DRAW);
		
	}

	void update() {

	}

	void render() {

		static simlod_gentree_cuda_nonprogressive::VoxelTreeGen* method = nullptr;
		if(method == nullptr){
			method = new simlod_gentree_cuda_nonprogressive::VoxelTreeGen(renderer, lasfile, numPoints);
		}

		// static simlod_gentree_cuda::VoxelTreeGen* method = nullptr;
		// if(method == nullptr){
		// 	method = new simlod_gentree_cuda::VoxelTreeGen(renderer, lasfile, ssPoints, numPoints);
		// }

		method->render();

	}


};

}
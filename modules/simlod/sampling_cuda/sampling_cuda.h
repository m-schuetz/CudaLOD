
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
#include "Shader.h"
#include "cudaGL.h"
#include "simlod/sampling_cuda/kernel_data.h.cu"
#include "simlod/LasLoader/LasLoader.h"
#include "Runtime.h"

#include "CudaModularProgram.h"



namespace simlod_gentree_cuda{

	// #define MAX_BUFFER_SIZE (1024 * 1024 * 1024)
	#define MAX_BUFFER_SIZE 2147483647

struct VoxelTreeGen{

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

	struct State{ 
		uint32_t semaphores[64];  // 64
		uint32_t numGuiItems;     //  4
		GuiItem guiItems[16];     // 64
		uint32_t wasJustCompiled;
		PointCloudMetadata metadata;
	};

	Renderer* renderer = nullptr;
	shared_ptr<simlod::LasFile> lasfile = nullptr;
	int numPoints = 0;

	GLuint ssIndirectPoints, ssIndirectLines, ssIndirectTriangles;
	GLuint ssBuffer;

	CUdeviceptr fb;
	CUgraphicsResource cur_input_points, cur_buffer, cur_points, cur_lines, cur_triangles;
	CUevent start, end;

	CudaModularProgram* cudaProgram = nullptr;

	Shader* sh_points = nullptr;
	Shader* sh_lines = nullptr;
	Shader* sh_triangles = nullptr;
	GLuint vao = -1;

	GLuint ssPoints = -1;

	int MAX_POINTS = 30'000'000;
	int MAX_LINES = 30'000'000;
	int MAX_TRIANGLES = 30'000'000;

	bool registered = false;

	VoxelTreeGen(Renderer* renderer, shared_ptr<simlod::LasFile> lasfile, GLuint ssPoints, int numPoints){

		this->renderer = renderer;
		this->lasfile = lasfile;
		this->ssPoints = ssPoints;
		this->numPoints = numPoints;

		sh_points = new Shader(
			"./modules/simlod/sampling_cuda/points.vs", 
			"./modules/simlod/sampling_cuda/points.fs"
		);

		sh_lines = new Shader(
			"./modules/simlod/sampling_cuda/lines.vs", 
			"./modules/simlod/sampling_cuda/lines.fs"
		);

		sh_triangles = new Shader(
			"./modules/simlod/sampling_cuda/triangles.vs", 
			"./modules/simlod/sampling_cuda/triangles.fs"
		);

		cuMemAlloc(&fb, 8 * 2048 * 2048);

		cudaProgram = new CudaModularProgram({
			.modules = {
				"./modules/simlod/sampling_cuda/lib.cu",
				"./modules/simlod/sampling_cuda/kernel_moduled.cu"
			},
			.kernels = {"kernel"}
		});

		cuEventCreate(&start, CU_EVENT_DEFAULT);
		cuEventCreate(&end, CU_EVENT_DEFAULT);

		glGenVertexArrays(1, &vao);

		glCreateBuffers(1, &ssBuffer);
		glCreateBuffers(1, &ssIndirectPoints);
		glCreateBuffers(1, &ssIndirectLines);
		glCreateBuffers(1, &ssIndirectTriangles);
		glNamedBufferData(ssBuffer, MAX_BUFFER_SIZE, nullptr, GL_DYNAMIC_DRAW);
		glNamedBufferData(ssIndirectPoints, 16 + 16 * MAX_POINTS, nullptr, GL_DYNAMIC_DRAW);
		glNamedBufferData(ssIndirectLines, 16 + 32 * MAX_LINES, nullptr, GL_DYNAMIC_DRAW);
		glNamedBufferData(ssIndirectTriangles, 16 + 48 * MAX_TRIANGLES, nullptr, GL_DYNAMIC_DRAW);

		voxelize(true);

		cudaProgram->onCompile([&](){
			Runtime::guiItems.clear();
			voxelize(true);
		});

	}

	~VoxelTreeGen(){
		// std::vector< CUgraphicsResource> persistent_resources = { batches, xyz12, xyz8, xyz4, rgba };
		// cuGraphicsUnmapResources(persistent_resources.size(), persistent_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
		// cuGraphicsUnregisterResource(rgba);
		// cuGraphicsUnregisterResource(batches);
		// cuGraphicsUnregisterResource(xyz12);
		// cuGraphicsUnregisterResource(xyz8);
		// cuGraphicsUnregisterResource(xyz4);
		// cuGraphicsUnregisterResource(output);
	}

	void voxelize(bool wasJustCompiled = false){
		if (cudaProgram == nullptr){
			return;
		}

		//cout << "wasJustCompiled: " << wasJustCompiled << endl;

		auto tStart = now();

		CUresult resultcode = CUDA_SUCCESS;

		cuGraphicsGLRegisterBuffer(&cur_input_points, ssPoints, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
		resultcode = cuGraphicsGLRegisterBuffer(&cur_buffer, ssBuffer, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
		cuGraphicsGLRegisterBuffer(&cur_points, ssIndirectPoints, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
		cuGraphicsGLRegisterBuffer(&cur_lines, ssIndirectLines, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
		cuGraphicsGLRegisterBuffer(&cur_triangles, ssIndirectTriangles, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);

		cout << endl;
		cout << "==== run cuda ===" << endl;

		std::vector<CUgraphicsResource> dynamic_resources = {cur_input_points, cur_buffer, cur_points, cur_lines, cur_triangles};
		cuGraphicsMapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

		static CUdeviceptr ptr_input_points, ptr_buffer, ptr_points, ptr_lines, ptr_triangles;
		size_t size;
		cuGraphicsResourceGetMappedPointer(&ptr_input_points, &size, cur_input_points);
		resultcode = cuGraphicsResourceGetMappedPointer(&ptr_buffer, &size, cur_buffer);
		cuGraphicsResourceGetMappedPointer(&ptr_points, &size, cur_points);
		cuGraphicsResourceGetMappedPointer(&ptr_lines, &size, cur_lines);
		cuGraphicsResourceGetMappedPointer(&ptr_triangles, &size, cur_triangles);

		CudaProgramSettings cdata;
		cdata.test = 345;

		// cuMemsetD8(ptr_buffer, 0x00, MAX_BUFFER_SIZE);
		resultcode = cuMemsetD8(ptr_buffer, 0x00, 1024);
		cuMemsetD8(ptr_points, 0x00, 16);
		cuMemsetD8(ptr_lines, 0x00, 16);
		cuMemsetD8(ptr_triangles, 0x00, 16);

		{ // initialize state data

			State state;
			memset(&state, 0, sizeof(state));

			state.metadata.numPoints = this->numPoints;
			state.metadata.min_x = lasfile->header.boxMin.x;
			state.metadata.min_y = lasfile->header.boxMin.y;
			state.metadata.min_z = lasfile->header.boxMin.z;
			state.metadata.max_x = lasfile->header.boxMax.x;
			state.metadata.max_y = lasfile->header.boxMax.y;
			state.metadata.max_z = lasfile->header.boxMax.z;

			// Buffer state(1024);

			for(int i = 0; i < Runtime::guiItems.size(); i++){
				auto guiItem = Runtime::guiItems[i];

				// int offset = 4 * 16 + 4 + (i * stride) + 12;
				// glNamedBufferSubData(ssBuffer, offset, sizeof(float), guiItem.value);
				// buffer.set<float>(offset, guiItem.value);
				state.guiItems[i].value = guiItem.value;
			}

			// buffer.set<bool>(offset, guiItem.value);+
			state.wasJustCompiled = wasJustCompiled ? 1 : 0;

			resultcode =  cuMemcpyHtoD(ptr_buffer, &state, sizeof(State));
		}

		void* args[] = {&cdata, &fb, &ptr_buffer, &ptr_input_points, &ptr_points, &ptr_lines, &ptr_triangles};
		// void* args[] = { &cdata, &fb, &batches_ptr, &xyz12_ptr, &xyz8_ptr, &xyz4_ptr, &rgba_ptr, &bbs_ptr };
		int numGroups = 50;
		int workgroupSize = 128;
		// resultcode =  cuLaunchKernel(cudaProgram->kernel,
		// 	numGroups, 1, 1,
		// 	workgroupSize, 1, 1,
		// 	0, 0, args, 0);

		//cuOccupancyMaxPotentialBlockSize(

		auto tLaunchStart = now();
		// resultcode =  cuLaunchKernel(cudaProgram->kernel,
		// 	numGroups, 1, 1,
		// 	workgroupSize, 1, 1,
		// 	0, 0, args, 0);

		auto res_launch = cuLaunchCooperativeKernel(cudaProgram->kernels["kernel"],
			numGroups, 1, 1,
			workgroupSize, 1, 1,
			0, 0, args);

		if(res_launch != CUDA_SUCCESS){
			const char* str; 
			cuGetErrorString(res_launch, &str);
			printf("error: %s \n", str);
		}

		printElapsedTime("launch: ", tLaunchStart);
		

		cuCtxSynchronize();

		Buffer result(1024);
		resultcode =  cuMemcpyDtoH(result.data, ptr_buffer, 1024);
		//cout << "CUresult: " << resultcode << endl;

		cuCtxSynchronize();

		Runtime::guiItems.clear();
		uint32_t numGuiItems  = result.get<uint32_t>(4 * 16 + 0);
		for(int i = 0; i < numGuiItems; i++){

			int stride = 16 + 32;
			Runtime::GuiItem item;
			item.type  = result.get<uint32_t>(4 * 16 + 4 + (i * stride) +  0);
			item.min   = result.get<float   >(4 * 16 + 4 + (i * stride) +  4);
			item.max   = result.get<float   >(4 * 16 + 4 + (i * stride) +  8);
			item.value = result.get<float   >(4 * 16 + 4 + (i * stride) + 12);
			item.oldValue = item.value;
			
			char label[32];
			memcpy(reinterpret_cast<void*>(&label[0]), result.data_u8 + 4 * 16 + 4 + (i * stride) + 16, 32);
			item.label = string(reinterpret_cast<const char*>(&label[0]), 32);

			Runtime::guiItems.push_back(item);
		}

		cuGraphicsUnmapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

		cuGraphicsUnregisterResource(cur_input_points);
		cuGraphicsUnregisterResource(cur_buffer);
		cuGraphicsUnregisterResource(cur_points);
		cuGraphicsUnregisterResource(cur_lines);
		cuGraphicsUnregisterResource(cur_triangles);

		auto tEnd = now();
		cout << "cuda duration: " << formatNumber(1000.0 * (tEnd - tStart), 1) << "ms" << endl;

	}

	void render(){


		bool guiValueChanged = false;
		for(auto& guiItem : Runtime::guiItems){
			if(guiItem.value != guiItem.oldValue){
				guiValueChanged = true;
			}
		}

		if(guiValueChanged){
			voxelize(false);
		}


		auto fbo = renderer->views[0].framebuffer;
		glBindFramebuffer(GL_FRAMEBUFFER, fbo->handle);

		glBindVertexArray(vao);

		auto view = renderer->camera->view;
		auto proj = renderer->camera->proj;
		mat4 viewProj = proj * view;

		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

		{ // POINTS
			glUseProgram(sh_points->program);

			glUniformMatrix4fv(0, 1, GL_FALSE, &viewProj[0][0]);

			glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, ssIndirectPoints, 16, 16 * MAX_POINTS);
			glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ssIndirectPoints);
			glDrawArraysIndirect(GL_POINTS, 0);

			// GLBuffer glbuffer;
			// glbuffer.handle = ssIndirectPoints;
			// auto buffer = renderer->readBuffer(glbuffer, 0, 128);
		}

		{ // LINES
			glUseProgram(sh_lines->program);
			
			glUniformMatrix4fv(0, 1, GL_FALSE, &viewProj[0][0]);

			glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, ssIndirectLines, 16, 16 * MAX_LINES);
			glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ssIndirectLines);
			glDrawArraysIndirect(GL_LINES, 0);
		}

		{ // TRIANGLES
			glUseProgram(sh_triangles->program);
			
			glUniformMatrix4fv(0, 1, GL_FALSE, &viewProj[0][0]);

			glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, ssIndirectTriangles, 16, 16 * MAX_TRIANGLES);
			glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ssIndirectTriangles);
			glDrawArraysIndirect(GL_TRIANGLES, 0);
		}
	}

};



};


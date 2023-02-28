
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
#include "simlod/LasLoader/LasLoader.h"



namespace simlod_gentree{

struct UniformData{
	int numPoints = 0;
	float min_x;
	float min_y;
	float min_z;
	float max_x;
	float max_y;
	float max_z;
	int gridSize;
};

struct VoxelTreeGen{

	Renderer* renderer = nullptr;
	shared_ptr<simlod::LasFile> lasfile = nullptr;
	GLuint ssPoints;
	GLuint ssIndirectPoints, ssIndirectLines;
	GLuint ssBuffer, ssState;
	int numPoints = 0;

	Shader* sh_points = nullptr;
	Shader* sh_lines = nullptr;
	GLuint vao = -1;

	int MAX_POINTS = 10'000'000;
	int MAX_LINES = 10'000'000;

	VoxelTreeGen(Renderer* renderer, shared_ptr<simlod::LasFile> lasfile, GLuint ssPoints, int numPoints){

		this->renderer = renderer;
		this->lasfile = lasfile;
		this->ssPoints = ssPoints;
		this->numPoints = numPoints;

		sh_points = new Shader(
			"./modules/simlod/voxel_sampling_gentree/points.vs", 
			"./modules/simlod/voxel_sampling_gentree/points.fs"
		);

		sh_lines = new Shader(
			"./modules/simlod/voxel_sampling_gentree/lines.vs", 
			"./modules/simlod/voxel_sampling_gentree/lines.fs"
		);

		glGenVertexArrays(1, &vao);

		glCreateBuffers(1, &ssBuffer);
		glCreateBuffers(1, &ssState);
		glCreateBuffers(1, &ssIndirectPoints);
		glCreateBuffers(1, &ssIndirectLines);
		glNamedBufferData(ssBuffer, 1024 * 1024 * 1024, nullptr, GL_DYNAMIC_DRAW);
		glNamedBufferData(ssState, 1024, nullptr, GL_DYNAMIC_DRAW);
		glNamedBufferData(ssIndirectPoints, 16 + 16 * MAX_POINTS, nullptr, GL_DYNAMIC_DRAW);
		glNamedBufferData(ssIndirectLines, 16 + 32 * MAX_LINES, nullptr, GL_DYNAMIC_DRAW);

		voxelize();

	}

	void voxelize(){
		static Shader* csTryInsert = nullptr;

		static GLBuffer ssUniformBuffer, ssDebug, ssOctree, ssNodesToSplit;

		static UniformData uniformData;
		
		if(csTryInsert == nullptr){
			csTryInsert = new Shader({ {"./modules/simlod/voxel_sampling_gentree/try_insert.cs", GL_COMPUTE_SHADER} });

			csTryInsert->onCompile([&](){
				voxelize();
			});
			
			ssOctree = renderer->createBuffer(1'000'000);
			ssNodesToSplit = renderer->createBuffer(1'000'000);
			ssUniformBuffer = renderer->createBuffer(1024);
			ssDebug = renderer->createBuffer(1024);
		}

		GLTimerQueries::timestampPrint("voxelize-start");

		GLTimerQueries::timestampPrint("voxelize-clear-start");
		GLuint zero = 0;
		glClearNamedBufferData(ssUniformBuffer.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(ssState, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		// glClearNamedBufferData(ssIndirectPoints, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		// glClearNamedBufferData(ssIndirectLines, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferSubData(ssIndirectPoints, GL_R32UI, 0, 16, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferSubData(ssIndirectLines, GL_R32UI, 0, 16, GL_RED, GL_UNSIGNED_INT, &zero);
		GLTimerQueries::timestampPrint("voxelize-clear-end");
		
		{ // Update Uniform Buffer
			auto boxSize = lasfile->header.boxMax - lasfile->header.boxMin;
			float size = std::max(std::max(boxSize.x, boxSize.y), boxSize.z);

			uniformData.numPoints = this->numPoints;
			uniformData.min_x = lasfile->header.boxMin.x;
			uniformData.min_y = lasfile->header.boxMin.y;
			uniformData.min_z = lasfile->header.boxMin.z;
			uniformData.max_x = lasfile->header.boxMin.x + size;
			uniformData.max_y = lasfile->header.boxMin.y + size;
			uniformData.max_z = lasfile->header.boxMin.z + size;

			glNamedBufferSubData(ssUniformBuffer.handle, 0, sizeof(UniformData), &uniformData);
		}

		{ // TRY INSERT
			GLTimerQueries::timestampPrint("voxelize-project-start");
			glUseProgram(csTryInsert->program);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  0, ssPoints);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  5, ssBuffer);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, ssIndirectPoints);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, ssIndirectLines);
			glBindBufferBase(GL_UNIFORM_BUFFER, 20, ssUniformBuffer.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 21, ssState);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
		
			// int numGroups = this->numPoints / 128;
			int numGroups = 84;
			glDispatchCompute(numGroups, 1, 1);
			GLTimerQueries::timestampPrint("voxelize-project-end");
		}

		{ // DEBUG
			auto buffer = renderer->readBuffer(ssDebug, 0, 1024);

			uint32_t value = buffer->get<uint32_t>(0);

			cout << "[gentree] debug.value: " << value << endl;
		}

		GLTimerQueries::timestampPrint("voxelize-end");

	}

	void render(){
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
		}

		{ // LINES
			glUseProgram(sh_lines->program);
			
			glUniformMatrix4fv(0, 1, GL_FALSE, &viewProj[0][0]);

			glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, ssIndirectLines, 16, 16 * MAX_LINES);
			glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ssIndirectLines);
			glDrawArraysIndirect(GL_LINES, 0);
		}
	}

};



};


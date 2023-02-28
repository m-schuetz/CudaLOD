

#include <iostream>
#include <filesystem>

#include "GLBuffer.h"
#include "Renderer.h"
#include "Shader.h"

#include "nvrtc.h"
#include "cuda.h"

#include "compute_voxel_rasterization/compute_voxel_rasterization.h"

#include "Runtime.h"

using namespace std;

int main(){

	cout << std::setprecision(2) << std::fixed;

	auto renderer = make_shared<Renderer>();

	// Creating a CUDA context
	cuInit(0);
	CUdevice cuDevice;
	CUcontext context;
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&context, 0, cuDevice);

	auto tStart = now();

	struct Setting{
		float yaw = 0.0;
		float pitch = 0.0;
		float radius = 0.0;
		dvec3 target;
	};
	
	unordered_map<string, Setting> settings;

	{// 
		Setting setting;
		setting.yaw = -0.15;
		setting.pitch = -0.57;
		setting.radius = 37.97;
		setting.target = {-0.99, -2.23, -3.68};

		
		setting.yaw = -1.81;
		setting.pitch = -0.19;
		setting.radius = 25.93;
		setting.target = {12.98, 8.40, 4.11};


		settings["default"] = setting;
	}

	Setting setting = settings["default"];
	
	renderer->controls->yaw = setting.yaw;
	renderer->controls->pitch = setting.pitch;
	renderer->controls->radius = setting.radius;
	renderer->controls->target = setting.target;

	GLTimerQueries::frameStart();
	auto parametric = make_shared<ComputeVoxelRasterization>(renderer.get());
	GLTimerQueries::frameEnd();

	auto update = [&](){
		parametric->update();
	};

	auto render = [&](){
		
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		{
			auto& view = renderer->views[0];

			view.view = renderer->camera->view;
			view.proj = renderer->camera->proj;

			renderer->views[0].framebuffer->setSize(renderer->width, renderer->height);

			glBindFramebuffer(GL_FRAMEBUFFER, view.framebuffer->handle);
			glClearColor(0.0, 0.2, 0.3, 1.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		}

		{
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			auto selected = Runtime::getSelectedMethod();
			if(selected){
				selected->render(renderer.get());
			}

			parametric->render();
		}

	};

	renderer->loop(update, render);

	return 0;
}


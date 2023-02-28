

#include <iostream>
#include <filesystem>

#include "GLBuffer.h"
#include "Renderer.h"
#include "Shader.h"
#include "compute_basic.h"
#include "ProgressiveFileBuffer.h"

#include "compute_loop_compress_nodewise/compute_loop_compress_nodewise.h"
#include "compute_loop_las/compute_loop_las.h"
#include "compute_loop_las2/compute_loop_las2.h"
#include "compute_loop_las_hqs/compute_loop_las_hqs.h"

#include "odlod/OnDemandLOD.h"

#include "Runtime.h"
#include "Method.h"
#include "compute/ComputeLasLoader.h"
#include "compute/LasLoaderStandard.h"
#include "compute/PotreeData.h"

#include "compute/LasLoaderSparse.h"

using namespace std;

int main(){

	cout << std::setprecision(2) << std::fixed;

	auto renderer = make_shared<Renderer>();

	auto tStart = now();

	struct Setting{
		string path_potree = "";
		string path_las = "";
		float yaw = 0.0;
		float pitch = 0.0;
		float radius = 0.0;
		dvec3 target;
	};

	Setting setting;
	// setting.yaw = 0.53;
	// setting.pitch = -0.68;
	// setting.radius = 614.70;
	// setting.target = { 0.0, 0.0, 0.0};

	renderer->controls->yaw = -0.19;
	renderer->controls->pitch = -0.88;
	renderer->controls->radius = 64987.93;
	renderer->controls->target = {-32600.67, 23548.88, 542.20};


	GLTimerQueries::frameStart();
	auto odlod = make_shared<odlod::OnDemandLOD>(setting.path_las, renderer.get());
	GLTimerQueries::frameEnd();

	auto lasLoaderSparse = make_shared<LasLoaderSparse>(renderer);

	Runtime::lasLoaderSparse = lasLoaderSparse;

	renderer->onFileDrop([lasLoaderSparse, renderer](vector<string> files){

		vector<string> lasfiles;

		for(auto file : files){
			if(iEndsWith(file, "las") || iEndsWith(file, "laz")){
				lasfiles.push_back(file);
			}
		}

		lasLoaderSparse->add(lasfiles, [renderer](vector<shared_ptr<LasFile>> lasfiles){

			dvec3 boxMin = {Infinity, Infinity, Infinity};
			dvec3 boxMax = {-Infinity, -Infinity, -Infinity};

			for(auto lasfile : lasfiles){
				boxMin = glm::min(boxMin, lasfile->boxMin);
				boxMax = glm::max(boxMax, lasfile->boxMax);
			}

			// zoom to point cloud
			auto size = boxMax - boxMin;
			auto position = (boxMax + boxMin) / 2.0;
			auto radius = glm::length(size) / 1.5;

			renderer->controls->yaw = 0.53;
			renderer->controls->pitch = -0.68;
			renderer->controls->radius = radius;
			renderer->controls->target = position;
		});

		glfwFocusWindow(renderer->window);
	});

	{ // 4-4-4 byte format
		auto computeLoopLas       = new ComputeLoopLas(renderer.get(), lasLoaderSparse);
		auto computeLoopLas2      = new ComputeLoopLas2(renderer.get(), lasLoaderSparse);
		auto computeLoopLasHqs    = new ComputeLoopLasHqs(renderer.get(), lasLoaderSparse);
		Runtime::addMethod((Method*)computeLoopLas);
		Runtime::addMethod((Method*)computeLoopLas2);
		Runtime::addMethod((Method*)computeLoopLasHqs);
	}

	auto update = [&](){

		lasLoaderSparse->process();

		auto selected = Runtime::getSelectedMethod();
		if(selected){

			bool needsVR = false;
			needsVR = needsVR || selected->name == "loop_las_hqs_vr";
			needsVR = needsVR || selected->name == "loop_nodes_hqs_vr";
			if(needsVR){
				renderer->setVR(true);
			}else{
				renderer->setVR(false);
			}
			

			selected->update(renderer.get());
		}

		odlod->update();
		
		if(Runtime::resource){

			string state = "";
			if(Runtime::resource->state == ResourceState::LOADED){
				state = "LOADED";
			}else if(Runtime::resource->state == ResourceState::LOADING){
				state = "LOADING";
			}else if(Runtime::resource->state == ResourceState::UNLOADED){
				state = "UNLOADED";
			}else if(Runtime::resource->state == ResourceState::UNLOADING){
				state = "UNLOADING";
			}
			

			Debug::set("state", state);
		}

		for(auto lasfile : Runtime::lasLoaderSparse->files){
			if(lasfile->isDoubleClicked){

				auto size = lasfile->boxMax - lasfile->boxMin;
				auto position = (lasfile->boxMax + lasfile->boxMin) / 2.0;
				auto radius = glm::length(size) / 1.5;

				renderer->controls->yaw = 0.53;
				renderer->controls->pitch = -0.68;
				renderer->controls->radius = radius;
				renderer->controls->target = position;
			}
		}

		// renderer->drawBoundingBox({0.0, 0.0, 0.0}, {200.0, 200.0, 200.0}, {200, 0, 0});

	};

	auto render = [&](){
		
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		if(renderer->vrEnabled){

			auto ovr = OpenVRHelper::instance();


			auto flip = glm::dmat4(
				1.0, 0.0, 0.0, 0.0,
				0.0, 0.0, 1.0, 0.0,
				0.0, -1.0, 0.0, 0.0,
				0.0, 0.0, 0.0, 1.0
			);

			auto& viewLeft = renderer->views[0];
			auto& viewRight = renderer->views[1];

			


			if(!Debug::dummyVR){
				auto size = ovr->getRecommmendedRenderTargetSize();
				viewLeft.framebuffer->setSize(size[0], size[1]);
				viewRight.framebuffer->setSize(size[0], size[1]);

				auto poseHMD = ovr->hmdPose;
				auto poseLeft = ovr->getEyePose(vr::Hmd_Eye::Eye_Left);
				auto poseRight = ovr->getEyePose(vr::Hmd_Eye::Eye_Right);

				viewLeft.view = glm::inverse(flip * poseHMD * poseLeft);
				viewLeft.proj = ovr->getProjection(vr::Hmd_Eye::Eye_Left, 0.01, 10'000.0);

				viewRight.view = glm::inverse(flip * poseHMD * poseRight);
				viewRight.proj = ovr->getProjection(vr::Hmd_Eye::Eye_Right, 0.01, 10'000.0);
			}else{
				ivec2 size = {2468, 2740};
				viewLeft.framebuffer->setSize(size[0], size[1]);
				viewRight.framebuffer->setSize(size[0], size[1]);
			}

			//viewLeft.framebuffer->setSize(1440, 1600);
			glBindFramebuffer(GL_FRAMEBUFFER, viewLeft.framebuffer->handle);
			glClearColor(0.8, 0.2, 0.3, 1.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			//viewRight.framebuffer->setSize(1440, 1600);
			glBindFramebuffer(GL_FRAMEBUFFER, viewRight.framebuffer->handle);
			glClearColor(0.0, 0.8, 0.3, 1.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		}else{

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

			odlod->render();
		}

		
		for(auto lasfile : Runtime::lasLoaderSparse->files){

			dvec3 size = lasfile->boxMax - lasfile->boxMin;
			dvec3 position = (lasfile->boxMax + lasfile->boxMin) / 2.0;

			if(lasfile->isHovered){
				renderer->drawBoundingBox(position, size, {255, 255, 0});
			}else if(lasfile->numPointsLoaded < lasfile->numPoints){
				renderer->drawBoundingBox(position, size, {255, 0, 0});
			}
			
		}
		

	};

	renderer->loop(update, render);

	return 0;
}


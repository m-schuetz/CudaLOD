

#include <iostream>
#include <filesystem>
#include<locale.h>

#include "GLBuffer.h"
#include "Renderer.h"
#include "Shader.h"
#include "compute_basic.h"
#include "ProgressiveFileBuffer.h"

#include "simlod/SimLOD.h"

//#include "VrRuntime.h"
#include "Runtime.h"
#include "Method.h"
#include "compute/ComputeLasLoader.h"
#include "compute/LasLoaderStandard.h"
#include "compute/PotreeData.h"

#include "compute/LasLoaderSparse.h"

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>
#include "glm/vec3.hpp"
#include "glm/vec2.hpp"




using namespace std;

int numPoints = 1'000'000;

int main(){

	cout << std::setprecision(2) << std::fixed;

	auto renderer = make_shared<Renderer>();
	//renderer->init();

	// Creating a CUDA context
	cuInit(0);
	CUdevice cuDevice;
	CUcontext context;
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&context, 0, cuDevice);

	{
		setlocale( LC_ALL, "en_AT.UTF-8" );
		// setlocale(LC_ALL, "");

	}

	auto tStart = now();

	struct Setting{
		string path_potree = "";
		string path_las = "";
		float yaw = 0.0;
		float pitch = 0.0;
		float radius = 0.0;
		dvec3 target;
	};

	struct Box{
		vec3 min;
		vec3 max;
	};

	auto readBox = [](string path){
		auto buffer_header = readBinaryFile(path, 0, 375);

		

		Box box;
		box.min.x = buffer_header->get<double>(187);
		box.min.y = buffer_header->get<double>(203);
		box.min.z = buffer_header->get<double>(219);
		box.max.x = buffer_header->get<double>(179);
		box.max.y = buffer_header->get<double>(195);
		box.max.z = buffer_header->get<double>(211);

		return box;
	};
	
	 unordered_map<string, Setting> settings;
	
	// RETZ
	settings["retz"] = {
		// .path_las = "E:/dev/pointclouds/riegl/Retz_Airborne_Terrestrial_Combined_1cm.las",
		.path_las = "D:/dev/pointclouds/Riegl/retz.las",
		.yaw = 6.91,
		.pitch = -0.78,
	};

	// LION
	settings["lion"] = {
		// .path_las = "E:/dev/pointclouds/mschuetz/lion.las",
		.path_las = "D:/dev/pointclouds/mschuetz/lion.las",
		.yaw = 6.74,
		.pitch = -0.48,
	};

	// HEIDENTOR
	settings["heidentor"] = {
		 .path_las = "D:/dev/pointclouds/archpro/heidentor.las",
		//.path_las = "E:/dev/pointclouds/archpro/heidentor.las",
		.yaw = 6.65,
		.pitch = -0.71,
	};
	
	// CA13
	settings["ca13_tile"] = {
		.path_las = "E:/dev/pointclouds/opentopography/CA13_moro_area/ot_35120D7314D_1_1.las",
		.yaw = 0.53,
		.pitch = -0.68,
		.radius = 614.70,
		.target = { 0.0, 0.0, 0.0},
	};

	// Eclepens
	settings["eclepens"] = {
		// .path_las = "E:/dev/pointclouds/pix4d/eclepens.las",
		.path_las = "D:/dev/pointclouds/pix4d/eclepens.las",
		.yaw = 6.65,
		.pitch = -0.71,
		.radius = 1109.77,
		.target = {514.05, 475.84, -156.43},
	};

	// Tires
	settings["tires"] = {
		.path_las = "E:/dev/pointclouds/tyres.las",
		// .yaw = 6.65,
		// .pitch = -0.71,
		// .radius = 1109.77,
		// .target = {514.05, 475.84, -156.43},
	};

	// Lifeboat
	settings["lifeboat"] = {
		.path_las = "D:/dev/pointclouds/weiss/pos8_lifeboats.las",
		.yaw = -1.91,
		.pitch = -0.32,
		.radius = 16.75,
		.target = {9.76, 7.62, 3.82},
	};

	// MORRO BAY
	settings["morrobay_morton"] = {
		.path_las = "F:/temp/wgtest/morro_bay/morro_bay_278M_morton.las",
		.yaw = -0.15,
		.pitch = -0.57,
		.radius = 3166.32,
		.target = {2239.05, 1713.63, -202.02},
	};

	// SaintRoman_0
	settings["SaintRoman_0"] = {
		.path_las = "D:/dev/pointclouds/iconem/SaintRoman_cleaned_1094M_202003_halved.las",
		.yaw = 2.11,
		.pitch = -0.54,
		.radius = 72.99,
		.target = {26.86, 68.42, -0.07},
	};

	// SaintRoman_1
	settings["SaintRoman_1"] = {
		.path_las = "D:/dev/pointclouds/iconem/SaintRoman_cleaned_1094M_202003_halved.las",
		.yaw = -0.90,
		.pitch = -0.24,
		.radius = 7.41,
		.target = {21.87, 67.85, 16.95},
	};

	// SaintRoman_2
	settings["SaintRoman_2"] = {
		.path_las = "D:/dev/pointclouds/iconem/SaintRoman_cleaned_1094M_202003_halved.las",
		.yaw = -0.71,
		.pitch = 0.05,
		.radius = 5.57,
		.target = {36.90, 75.39, 17.30},
	};

	// Palmyra_BelTemple_0
	settings["Palmyra_BelTemple_0"] = {
		.path_las = "D:/dev/pointclouds/iconem/Palmyra_2022rework_BelTemple_2016.las",
		.yaw = 0.90,
		.pitch = -0.72,
		.radius = 319.32,
		.target = {174.29, 129.37, -44.03},
	};

	// Palmyra_BelTemple_1
	settings["Palmyra_BelTemple_1"] = {
		.path_las = "D:/dev/pointclouds/iconem/Palmyra_2022rework_BelTemple_2016.las",
		.yaw = 1.50,
		.pitch = 0.18,
		.radius = 24.36,
		.target = {164.82, 158.33, 24.33},
	};

	// CA21_Bunds
	settings["CA21_Bunds"] = {
		.path_las = "D:/dev/pointclouds/iconem/CA21_bunds_9_to_14.las",
	};
	
	// TU Projekt
	settings["tu_projekt"] = {
		.path_las = "D:/dev/pointclouds/tu_project/tu_projekt_every_2nd.las",
		//.yaw = 1.50,
		//.pitch = 0.18,
		//.radius = 24.36,
		//.target = {164.82, 158.33, 24.33},
	};

	// CA13_two_tiles
	settings["CA13_two_tiles"] = {
		.path_las = "D:/dev/pointclouds/ca13_2_tiles.las",
		//.yaw = 1.50,
		//.pitch = 0.18,
		//.radius = 24.36,
		//.target = {164.82, 158.33, 24.33},
	};

	// testdata
	settings["test_checkerboard"] = {
		.path_las = "D:/dev/pointclouds/cudalod_evaluation/generated.las",
		//.yaw = 1.50,
		//.pitch = 0.18,
		//.radius = 24.36,
		//.target = {164.82, 158.33, 24.33},
	};



	{// WHATEVER
		Setting setting;
		// setting.path_las = "F:/temp/wgtest/morro_bay/morro_bay_278M_morton.las"; 
		setting.path_las = "D:/dev/pointclouds/tuwien_baugeschichte/candi Sari/photoscan19_exterior - POLYDATA - photoscan19_exterior.las"; 
		setting.path_las = "D:/dev/pointclouds/sitn/mobile_2561500_1204500.las"; 
		setting.path_las = "D:/dev/pointclouds/Riegl/subset.las"; 

		settings["whatever"] = setting;
	}

	// PICK POINT CLOUD TO PROCESS
	// lion, eclepens_morton
	// Setting setting = settings["lion"];
	// Setting setting = settings["SaintRoman_1"];
	// Setting setting = settings["SaintRoman_2"];
	// Setting setting = settings["lion"];
	//  Setting setting = settings["Palmyra_BelTemple_0"];
	// Setting setting = settings["SaintRoman_2"];
	Setting setting = settings["retz"];

	if(setting.radius == 0.0)
	{ // override camera pos with sensible default
		auto box = readBox(setting.path_las);
		setting.yaw = 2.13;
		setting.pitch = -0.59;
		setting.radius = length(box.max - box.min);
		setting.target = (box.max - box.min) / 2.0f;
	}

	//setting.yaw = 6.88;
	//setting.pitch = -0.71;
	//setting.radius = 93.12;
	//setting.target = {907.56, 112.48, -8.18};



	renderer->controls->yaw = setting.yaw;
	renderer->controls->pitch = setting.pitch;
	renderer->controls->radius = setting.radius;
	renderer->controls->target = setting.target;

	GLTimerQueries::frameStart();
	auto simlod = make_shared<simlod::SimLOD>(setting.path_las, renderer.get());
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

	auto update = [&](){

		lasLoaderSparse->process();

		simlod->update();
		
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
				glm::ivec2 size = {2468, 2740};
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

			simlod->render();
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


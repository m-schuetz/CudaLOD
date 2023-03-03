
#pragma once

#include <string>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>
#include <format>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "unsuck.hpp"

#include "Box.h"
#include "Debug.h"
#include "Camera.h"
// #include "LasLoader.h"
#include "Frustum.h"
#include "Renderer.h"
#include "Shader.h"
#include "cudaGL.h"
// #include "simlod/LasLoader/LasLoader.h"
#include "Runtime.h"

#include "builtin_types.h"

#include "CudaModularProgram.h"
#include "common.h"

#include "utils.h"
#include "OctreeWriter.h"

#include <thrust/sort.h>
#include <thrust/functional.h>



namespace simlod_gentree_cuda_nonprogressive{

	// #define MAX_BUFFER_SIZE (1024 * 1024 * 1024)
	// #define MAX_BUFFER_SIZE 2'147'483'647

	// For Saint Roman with 547M points
	// #define MAX_BUFFER_SIZE 15'000'000'000

	// For Bernhard and the CA21 Bunds datas set with 975M points
	#define MAX_BUFFER_SIZE 5'000'000'000

struct VoxelTreeGen{

	Renderer* renderer = nullptr;
	shared_ptr<simlod::LasFile> lasfile = nullptr;
	int numPoints = 0;

	CUdeviceptr num_nodes, nodes, sorted, alloc_offset;
	CUdeviceptr ptr_points, ptr_lines;

	CUdeviceptr ptr_buffer, ptr_render_buffer, ptr_input_points;
	CUdeviceptr ptr_results;
	// CUevent start, end, start_split, end_split, start_voxelize, end_voxelize;
	CUgraphicsResource gl_framebuffer;

	CudaModularProgram*  prog_build_lod = nullptr;
	CudaModularProgram*  prog_render = nullptr;

	GLuint vao = -1;

	bool registered = false;

	VoxelTreeGen(Renderer* renderer, shared_ptr<simlod::LasFile> lasfile, int numPoints){

		this->renderer = renderer;
		this->lasfile = lasfile;
		this->numPoints = numPoints;

		cuMemAlloc(&ptr_buffer, MAX_BUFFER_SIZE);
		cuMemAlloc(&ptr_results, sizeof(Results));
		cuMemAlloc(&ptr_render_buffer, 100'000'000);
		cuMemAlloc(&ptr_input_points, 16 * lasfile->points.size());
		cuMemcpyHtoD(ptr_input_points, &lasfile->points[0], 16 * lasfile->points.size());

		prog_build_lod = new CudaModularProgram({
			.modules = {
				"./modules/simlod/sampling_cuda_nonprogressive/lib.cu",
				"./modules/simlod/sampling_cuda_nonprogressive/kernel.cu",
			},
			.kernels = {"kernel2", "kernel3"}
		});

		prog_render = new CudaModularProgram({
			.modules = {
				"./modules/simlod/sampling_cuda_nonprogressive/lib.cu",
				"./modules/simlod/sampling_cuda_nonprogressive/render.cu",
			},
			.kernels = {"kernel"},
		});

		cuMemAlloc(&num_nodes, sizeof(uint32_t));
		cuMemAlloc(&nodes, sizeof(void*));
		cuMemAlloc(&sorted, sizeof(void*));
		cuMemAlloc(&alloc_offset, sizeof(uint64_t));
		cuMemAlloc(&ptr_lines, sizeof(void*));
		cuMemAlloc(&ptr_points, sizeof(void*));

		// cuEventCreate(&start, CU_EVENT_DEFAULT);
		// cuEventCreate(&end, CU_EVENT_DEFAULT);
		// cuEventCreate(&start_split, CU_EVENT_DEFAULT);
		// cuEventCreate(&end_split, CU_EVENT_DEFAULT);
		// cuEventCreate(&start_voxelize, CU_EVENT_DEFAULT);
		// cuEventCreate(&end_voxelize, CU_EVENT_DEFAULT);

		glGenVertexArrays(1, &vao);

		voxelize(true);

		prog_build_lod->onCompile([&](){
			voxelize(true);
		});

		renderer->onUpdate([&](){
			if(Runtime::requestLodGeneration){
				cout << "update with strategy " << Runtime::samplingStrategy << " requested" << endl;
				Runtime::requestLodGeneration = false;
				voxelize(true);
			}
		});

	}

	~VoxelTreeGen(){

	}

	void voxelize(bool wasJustCompiled = false){
		if (prog_build_lod == nullptr){
			return;
		}

		auto tStart = now();

		CUresult resultcode = CUDA_SUCCESS;

		cout << endl;
		cout << "==== run cuda ===" << endl;

		State state;
		{
			state.metadata.numPoints = this->numPoints;
			state.metadata.min_x = lasfile->header.boxMin.x;
			state.metadata.min_y = lasfile->header.boxMin.y;
			state.metadata.min_z = lasfile->header.boxMin.z;
			state.metadata.max_x = lasfile->header.boxMax.x;
			state.metadata.max_y = lasfile->header.boxMax.y;
			state.metadata.max_z = lasfile->header.boxMax.z;

			auto& viewLeft = renderer->views[0];
			mat4 world;
			mat4 view = viewLeft.view;
			mat4 proj = viewLeft.proj;
			mat4 worldView = view * world;
			mat4 viewProj = mat4(proj) * view;
			mat4 worldViewProj = proj * view * world;

			auto fbo = renderer->views[0].framebuffer;
			*((glm::mat4*)&state.transform) = glm::transpose(worldViewProj);
			state.imageSize = int2{ fbo->width, fbo->height };

			state.strategy = static_cast<SamplingStrategy>(Runtime::samplingStrategy);
			state.LOD = Runtime::LOD;
		}

		void* args[] = {
			&state, &ptr_buffer, &ptr_results, &ptr_input_points, 
			&nodes, &num_nodes, &sorted, &alloc_offset,
			&ptr_points, &ptr_lines
		};
		
		CUevent start, end, start_split, end_split, start_voxelize, end_voxelize;
		cuEventCreate(&start, 0);
		cuEventCreate(&end, 0);
		cuEventCreate(&start_split, 0);
		cuEventCreate(&end_split, 0);
		cuEventCreate(&start_voxelize, 0);
		cuEventCreate(&end_voxelize, 0);

		// CUevent start, end;
		// cuEventCreate(&start, 0);
		// cuEventCreate(&end, 0);
		CUdevice device;
		int SMs;
		cuCtxGetDevice(&device);
		cuDeviceGetAttribute(&SMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
		
		auto tLaunchStart = now();

		{ // KERNEL 2
			int workgroupSize = 256;

			int maxActiveBlocksPerSM;
			cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, prog_build_lod->kernels["kernel2"], workgroupSize, 0);
			
			int numGroups = maxActiveBlocksPerSM * SMs;

			cout << std::format("launching kernel 2, groups: {}, groupSize: {} \n", numGroups, workgroupSize);
			cout << std::format("(note: maxGroupsPerSM: {} for workgroupSize {}, SMs: {} \n", maxActiveBlocksPerSM, workgroupSize, SMs);

			cuEventRecord(start, 0);
			cuEventRecord(start_split, 0);
			auto res_launch = cuLaunchCooperativeKernel(prog_build_lod->kernels["kernel2"],
				numGroups, 1, 1,
				workgroupSize, 1, 1,
				0, 0, args);

			if(res_launch != CUDA_SUCCESS){
				const char* str; 
				cuGetErrorString(res_launch, &str);
				printf("error: %s \n", str);
			}

			cuEventRecord(end_split, 0);
		}

		{ // KERNEL 3
			int workgroupSize = 256;
			int maxActiveBlocksPerSM;
			auto curesult = cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, 
				prog_build_lod->kernels["kernel3"], workgroupSize, 0);
			// numGroups *= SMs;

			// actually, let's set this to the amount of SMs
			// because some kernels allocate global memory for 
			// sampling grids for each workgroup/block
			int numGroups = SMs;
			// numGroups = 1;

			cout << "curesult: " << curesult << endl;
			cout << std::format("launching kernel 3, groups: {}, groupSize: {} \n", numGroups, workgroupSize);
			cout << std::format("(note: maxGroupsPerSM: {} for workgroupSize {}, SMs: {} \n", maxActiveBlocksPerSM, workgroupSize, SMs);

			// int dynamicSharedMemSize = 60'000;
			// int dynamicSharedMemSize = 98304;
			// cuFuncSetAttribute (prog_build_lod->kernels["kernel3"], 
			// 	CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamicSharedMemSize);

			cuEventRecord(start_voxelize, 0);

			auto res_launch2 = cuLaunchCooperativeKernel(prog_build_lod->kernels["kernel3"],
				numGroups, 1, 1,
				workgroupSize, 1, 1,
				0, 0, args);
			// auto res_launch2 = cuLaunchCooperativeKernel(prog_build_lod->kernels["kernel3"],
			// 	numGroups, 1, 1,
			// 	workgroupSize, 1, 1,
			// 	dynamicSharedMemSize, 0, args);

			if (res_launch2 != CUDA_SUCCESS) {
				const char* str;
				cuGetErrorString(res_launch2, &str);
				printf("error: %s \n", str);
			}

			cuEventRecord(end_voxelize, 0);
		}

		cuEventRecord(end, 0);
		cuEventSynchronize(end);

		float total_ms, split_ms, voxelize_ms;
		{
			cuEventElapsedTime(&total_ms, start, end);
			cuEventElapsedTime(&split_ms, start_split, end_split);
			cuEventElapsedTime(&voxelize_ms, start_voxelize, end_voxelize);

			cout << "CUDA durations: " << endl;
			cout << std::format("split:     {:6.1f} ms", split_ms) << endl;
			cout << std::format("voxelize:  {:6.1f} ms", voxelize_ms) << endl;
			cout << std::format("total:     {:6.1f} ms", total_ms) << endl;
		}

		cuCtxSynchronize();

		auto tEnd = now();
		// cout << "cuda duration: " << formatNumber(1000.0 * (tEnd - tStart), 1) << "ms" << endl;

		// if(false)
		{ // write test results

			// struct Point{
			// 	float x;
			// 	float y;
			// 	float z;
			// 	unsigned int color;
			// };
			// struct Node{
			// 	int pointOffset;
			// 	int numPoints;
			// 	Point* points;
			// 	int numAdded;
			// 	int level;
			// 	int voxelIndex;
			// 	vec3 min;
			// 	vec3 max;
			// 	float cubeSize;
			// 	Node* children[8];
			// 	int numVoxels = 0;
			// 	Point* voxels = nullptr;
			// 	bool visible = true;
			// };

			cout << "copy device to host" << endl;
			Buffer buffer(MAX_BUFFER_SIZE);
			cuMemcpyDtoH(buffer.data, ptr_buffer, buffer.size);

			uint64_t ptrNodes = 0;
			cuMemcpyDtoH(&ptrNodes, nodes, 8);

			uint32_t numNodes = 0;
			cuMemcpyDtoH(&numNodes, num_nodes, 4);

			cuCtxSynchronize();

			Box box;
			box.min = lasfile->header.boxMin;
			box.max = lasfile->header.boxMax;
			string path = "G:/temp/heidentor_weighted";
			OctreeWriter writer(path, box, &buffer, numNodes, ptr_buffer, ptrNodes);
			writer.write();
		}

		{ // RESULTS


			cout << "read results from device to host" << endl;
			Results results;
			cuMemcpyDtoH(&results, ptr_results, sizeof(Results));

			cuCtxSynchronize();

			stringstream ss;

			ss << "==== RESULTS ====" << endl;

			double pointsPerMS = double(results.points) / double(total_ms);
			uint64_t mpointsPerS = (pointsPerMS * 1000.0) / 1'000'000.0;
			
			auto locale = std::locale("en_GB.UTF-8");
			ss << std::format(locale, "#points:                     {:15L}", results.points) << endl;
			ss << std::format(locale, "#voxels:                     {:15L}", results.voxels) << endl;
			ss << std::format(locale, "#nodes:                      {:15L}", results.nodes) << endl;
			ss << std::format(locale, "million points / sec:        {:15L}", mpointsPerS) << endl;
			ss << std::format(locale, "#allocated (splitting):      {:15L}", results.allocatedMemory_splitting) << endl;
			ss << std::format(locale, "#allocated (voxelization):   {:15L}", results.allocatedMemory_voxelization) << endl;
			ss << std::format(locale, "min-avg-max points/node      {:7L} - {:7L} - {:7L}", 
				results.minPoints, results.avgPoints, results.maxPoints) << endl;
			ss << std::format(locale, "min-avg-max voxels/node      {:7L} - {:7L} - {:7L}", 
				results.minVoxels, results.avgVoxels, results.maxVoxels) << endl;

			ss << "histogram - points: " << endl;
			for(int i = 0; i < 25; i++){
				int sum = 0;
				sum += results.histogram_points[4 * i + 0];
				sum += results.histogram_points[4 * i + 1];
				sum += results.histogram_points[4 * i + 2];
				sum += results.histogram_points[4 * i + 3];
				ss << std::format("{:4},", sum);
			}
			ss << endl;
			ss << "histogram - voxels: " << endl;
			for(int i = 0; i < 25; i++){
				int sum = 0;
				sum += results.histogram_voxels[4 * i + 0];
				sum += results.histogram_voxels[4 * i + 1];
				sum += results.histogram_voxels[4 * i + 2];
				sum += results.histogram_voxels[4 * i + 3];
				cout << std::format("{:4},", sum);
			}
			ss << endl;

			ss << "level    inner     leaves         #points         #voxels       rate" << endl;
			for(int i = 0; i < 20; i++){
				
				int numPoints = results.pointsPerLevel[i];
				int numVoxels = results.voxelsPerLevel[i];
				int samples_0 = numPoints + numVoxels;
				int samples_1 = results.pointsPerLevel[i + 1] + results.voxelsPerLevel[i + 1];
				int numInnerNodes = results.innerNodesAtLevel[i];
				int numLeafNodes  = results.leafNodesAtLevel[i];

				if(samples_0 > 0){
					
					ss << std::format(locale, "{:2}: {:10L} {:10L} {:15L} {:15L}", 
						i, numInnerNodes, numLeafNodes, numPoints, numVoxels);

					if(samples_1 > 0){
						float percentage = float(samples_0) / float(samples_1);
						int percentage_i = 100.0f * percentage;

						// printf("   %5i %%", percentage_i);
						cout << std::format("      {:5} %", percentage_i);
					}

					ss << endl;
				}
			}

			cout << ss.str() << endl;


			stringstream ssHistogram_points;
			stringstream ssHistogram_voxels;

			for(int i = 0; i < 100; i++){
				ssHistogram_points << results.histogram_points[i] << ", ";
				ssHistogram_voxels << results.histogram_voxels[i] << ", ";
			}

			string strLevels = "";
			for(int i = 0; i < 20; i++){

				int numPoints = results.pointsPerLevel[i];
				int numVoxels = results.voxelsPerLevel[i];
				int samples_0 = numPoints + numVoxels;
				int samples_1 = results.pointsPerLevel[i + 1] + results.voxelsPerLevel[i + 1];
				int numInnerNodes = results.innerNodesAtLevel[i];
				int numLeafNodes  = results.leafNodesAtLevel[i];

				if(samples_0 > 0){

					strLevels += std::format("				[{:2}, {:5}, {:5}, {:9}, {:9}],\n", 
						i, numInnerNodes, numLeafNodes, numPoints, numVoxels);
				}
			}

			string strategy;
			if(results.strategy == SamplingStrategy::FIRST_COME){
				strategy = "FIRST_COME";
			}else if(results.strategy == SamplingStrategy::RANDOM){
				strategy = "RANDOM";
			}else if(results.strategy == SamplingStrategy::AVERAGE_SINGLECELL){
				strategy = "AVERAGE_SINGLECELL";
			}else if(results.strategy == SamplingStrategy::WEIGHTED_NEIGHBORHOOD){
				strategy = "WEIGHTED_NEIGHBORHOOD";
			}

			auto timestamp = std::chrono::system_clock::now();
			auto timestamp_t = std::chrono::system_clock::to_time_t(timestamp);
			// string datestring = std::ctime(&end_time);
			auto gmt_time = gmtime(&timestamp_t);
			stringstream sstime;
			sstime << std::put_time(gmt_time, "%Y-%m-%d %H:%M:%S");
			string datestring = sstime.str();


			cudaDeviceProp props;
			cudaGetDeviceProperties(&props, 0);
			string devicename = props.name;


			string formatString = R"V0G0N(
		{{
			"datetime": "{}",
			"device": "{}",
			"strategy": "{}",
			// duration in milliseconds
			"duration_split": {},
			"duration_voxelize": {},
			"duration_total": {},
			"points": {},
			"voxels": {},
			"nodes": {},
			// in million points per second
			"throughput": {},
			// allocated bytes after splitting and after voxelization
			"allocated_splitting": {},
			"allocated_voxelization": {},
			"minAvgMax_points": {},
			"minAvgMax_voxels": {},
			// histogram of number of nodes with certain amount of points or voxels
			"histogram_maxval": {},
			"histogram_points": [{}],
			"histogram_voxels": [{}],
			// [<level>, <#innerNodes>, <#leafNodes>, <#points>, <#voxels>]
			"levels": [
{}			],
		}})V0G0N";

			string strMinMax_points = std::format("[{:5}, {:5}, {:5}]", results.minPoints, results.avgPoints, results.maxPoints);
			string strMinMax_voxels = std::format("[{:5}, {:5}, {:5}]", results.minVoxels, results.avgVoxels, results.maxVoxels);

			string formatted = std::format(formatString, 
				datestring, devicename, strategy, 
				split_ms, voxelize_ms, total_ms,
				results.points, results.voxels, results.nodes, mpointsPerS, 
				results.allocatedMemory_splitting, results.allocatedMemory_voxelization,
				strMinMax_points, strMinMax_voxels,
				results.histogram_maxval, ssHistogram_points.str(), ssHistogram_voxels.str(),
				strLevels
			);

			static bool initialized = false;
			static stringstream ssResults;

			if(!initialized){
				ssResults << "{" << endl;
				ssResults << "	\"benchmarks\": [" << endl;
				initialized = true;
			}

			ssResults << formatted << ", " << endl;
			
			ofstream stream;
			stream.open("./results.json", ios::out);
			stream << ssResults.str() << endl;
			stream << "	]" << endl;
			stream << "}" << endl;

		}
	}

	void render(){

		auto fbo = renderer->views[0].framebuffer;
		glBindFramebuffer(GL_FRAMEBUFFER, fbo->handle);

		glBindVertexArray(vao);

		auto view = renderer->camera->view;
		auto proj = renderer->camera->proj;
		mat4 viewProj = proj * view;

		// if(false)
		if(prog_render){

			cuCtxSynchronize();

			static bool registered = false;
			if(!registered){
				cuGraphicsGLRegisterImage(
					&gl_framebuffer, 
					renderer->views[0].framebuffer->colorAttachments[0]->handle, 
					GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD
				);

				registered = true;
			}
			
			std::vector<CUgraphicsResource> dynamic_resources = {gl_framebuffer};
			cuGraphicsMapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

			CUDA_RESOURCE_DESC res_desc = {
				.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY,
			};
			cuGraphicsSubResourceGetMappedArray(&res_desc.res.array.hArray, gl_framebuffer, 0, 0);
			CUsurfObject output_surface;
			cuSurfObjectCreate(&output_surface, &res_desc);
			
			auto& viewLeft = renderer->views[0];
			mat4 world;
			mat4 view = viewLeft.view;
			mat4 proj = viewLeft.proj;
			mat4 worldViewProj = proj * view * world;

			struct RenderPassArgs{
				Mat4 transform;
				int2 imageSize;
				int numPoints;
				float LOD;
			};

			auto fbo = renderer->views[0].framebuffer;
			RenderPassArgs rpArgs;
			*((glm::mat4*)&rpArgs.transform) = glm::transpose(worldViewProj);
			rpArgs.imageSize = int2{ fbo->width, fbo->height };
			rpArgs.numPoints = lasfile->points.size();
			rpArgs.LOD = Runtime::LOD;

			void* args[] = {
				&rpArgs, &output_surface,
				&ptr_render_buffer,
				&nodes, &num_nodes, &sorted,
				&ptr_points, &ptr_lines
			};

			int numGroups = 80;
			int workgroupSize = 256;
			auto kernel = prog_render->kernels["kernel"];
			auto res_launch = cuLaunchCooperativeKernel(kernel,
				numGroups, 1, 1,
				workgroupSize, 1, 1,
				0, 0, args);

			if(res_launch != CUDA_SUCCESS){
				const char* str; 
				cuGetErrorString(res_launch, &str);
				printf("error: %s \n", str);
			}

			cuCtxSynchronize();

			cuGraphicsUnmapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
		}
	}

};



};


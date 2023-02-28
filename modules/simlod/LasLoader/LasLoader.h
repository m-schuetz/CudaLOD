
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
#include "LasLoader.h"
#include "Frustum.h"
#include "Renderer.h"

namespace simlod{


struct Point{
	float x;
	float y;
	float z;
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t a;
};

struct LasHeader{
	int64_t numPoints = 0;
	int64_t numPointsLoaded = 0;
	uint32_t offsetToPointData = 0;
	int pointFormat = 0;
	uint32_t bytesPerPoint = 0;
	dvec3 scale = {1.0, 1.0, 1.0};
	dvec3 offset = {0.0, 0.0, 0.0};
	dvec3 boxMin;
	dvec3 boxMax;
};

struct LasFile{
	string path;
	LasHeader header;
	vector<Point> points;
};

shared_ptr<LasFile> loadLas(string path){

	LasHeader header;

	{ // LOAD HEADER
		auto buffer_header = readBinaryFile(path, 0, 375);

		int versionMajor = buffer_header->get<uint8_t>(24);
		int versionMinor = buffer_header->get<uint8_t>(25);

		if(versionMajor == 1 && versionMinor < 4){
			header.numPoints = buffer_header->get<uint32_t>(107);
		}else{
			header.numPoints = buffer_header->get<uint64_t>(247);
		}

		header.numPoints = min(header.numPoints, 1'000'000'000ll);

		header.offsetToPointData = buffer_header->get<uint32_t>(96);
		header.pointFormat = buffer_header->get<uint8_t>(104) % 128;
		header.bytesPerPoint = buffer_header->get<uint16_t>(105);
		
		header.scale.x = buffer_header->get<double>(131);
		header.scale.y = buffer_header->get<double>(139);
		header.scale.z = buffer_header->get<double>(147);
		
		header.offset.x = buffer_header->get<double>(155);
		header.offset.y = buffer_header->get<double>(163);
		header.offset.z = buffer_header->get<double>(171);
		
		header.boxMin.x = buffer_header->get<double>(187);
		header.boxMin.y = buffer_header->get<double>(203);
		header.boxMin.z = buffer_header->get<double>(219);
		
		header.boxMax.x = buffer_header->get<double>(179);
		header.boxMax.y = buffer_header->get<double>(195);
		header.boxMax.z = buffer_header->get<double>(211);
	}

	auto lasfile = make_shared<LasFile>();
	lasfile->path = path;
	lasfile->header = header;

	{
		int64_t start = header.offsetToPointData;
		int64_t size = header.numPoints * header.bytesPerPoint;
		
		auto buffer = readBinaryFile(path, start, size);

		int64_t stride = header.bytesPerPoint;

		int rgbOffset = 0;
		if(header.pointFormat == 2) rgbOffset = 20;
		else if(header.pointFormat == 3) rgbOffset = 28;
		else if(header.pointFormat == 5) rgbOffset = 28;
		else if(header.pointFormat == 7) rgbOffset = 30;

		for(int i = 0; i < header.numPoints; i++){
			int32_t X = buffer->get<int32_t>(i * stride + 0);
			int32_t Y = buffer->get<int32_t>(i * stride + 4);
			int32_t Z = buffer->get<int32_t>(i * stride + 8);

			Point point;
			point.x = double(X) * header.scale.x + header.offset.x - header.boxMin.x;
			point.y = double(Y) * header.scale.y + header.offset.y - header.boxMin.y;
			point.z = double(Z) * header.scale.z + header.offset.z - header.boxMin.z;

			auto R = buffer->get<uint16_t>(i * stride + rgbOffset + 0);
			auto G = buffer->get<uint16_t>(i * stride + rgbOffset + 2);
			auto B = buffer->get<uint16_t>(i * stride + rgbOffset + 4);
			
			point.r = R > 255 ? R / 256 : R;
			point.g = G > 255 ? G / 256 : G;
			point.b = B > 255 ? B / 256 : B;
			point.a = 0;

			lasfile->points.push_back(point);

			
			
			if((i % 10'000'000) == 0){
				
				auto locale = std::locale("en_GB.UTF-8");
				cout << std::format(locale, "loaded {:L} points", i) << endl;
			}
		}

		auto locale = std::locale("en_GB.UTF-8");
		cout << std::format(locale, "finished loading {:L} points", lasfile->points.size()) << endl;

	}

	return lasfile;
}

};
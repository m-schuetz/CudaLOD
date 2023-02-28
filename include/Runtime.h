
#pragma once

#include <string>
#include <unordered_map>
#include <map>

#include "glm/common.hpp"

#include "Method.h"
#include "compute/Resources.h"

#include "compute/LasLoaderSparse.h"

using namespace std;



struct Runtime{

	struct GuiItem{
		uint32_t type = 0;
		float min = 0.0;
		float max = 1.0;
		float oldValue = 0.5;
		float value = 0.5;
		string label = "";
	};

	inline static vector<Method*> methods;
	inline static Method* selectedMethod = nullptr;
	inline static Resource* resource = nullptr;
	inline static vector<int> keyStates = vector<int>(65536, 0);
	inline static glm::dvec2 mousePosition = {0.0, 0.0};
	inline static int mouseButtons = 0;
	inline static shared_ptr<LasLoaderSparse> lasLoaderSparse = nullptr;
	inline static vector<GuiItem> guiItems;
	inline static int samplingStrategy = 0;
	inline static bool requestLodGeneration = false;


	Runtime(){
		
	}

	static Runtime* getInstance(){
		static Runtime* instance = new Runtime();

		return instance;
	}

	static void addMethod(Method* method){

		Runtime::methods.push_back(method);

	}

	static void setSelectedMethod(string name){
		for(Method* method : Runtime::methods){
			if(method->name == name){
				Runtime::selectedMethod = method;
			}
		}
	}

	static Method* getSelectedMethod(){
		return Runtime::selectedMethod;
	}

};
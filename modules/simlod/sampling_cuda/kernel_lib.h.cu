
#pragma once

#define FALSE 0
#define TRUE 1

#define LARGE_FLOAT 123000000000.0
#define LARGE_INT   123000000

typedef unsigned int uint32_t;
typedef int int32_t;
typedef char int8_t;
typedef unsigned char uint8_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

#define Infinity 0x7f800000

struct Point{
	float x;
	float y;
	float z;
	unsigned int color;
};

struct Points{
	unsigned int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	Point points[10 * 1000 * 1000];
};

struct Lines{
	unsigned int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	Point vertices[1000 * 1000];
};

struct Triangles{
	unsigned int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	Point vertices[1000000];
};

int test();

struct vec3{

	float x, y, z;

	vec3();
	vec3(float x, float y, float z);

	float length();

	float longestAxis();

	friend vec3 operator - (const vec3 &vector, float scalar);
	friend vec3 operator - (const vec3 &vector, vec3 other);
	friend vec3 operator + (const vec3 &vector, vec3 other);
	friend vec3 operator + (const vec3 &vector, float scalar);
	friend vec3 operator / (const vec3 &vector, float scalar);
	friend vec3 operator * (const vec3 &vector, float scalar);
};

struct Box3{
	vec3 min = {Infinity, Infinity, Infinity};
	vec3 max = {-Infinity, -Infinity, Infinity};

	Box3(){}

	vec3 center();
	vec3 size();
};

struct TimerItem{
	const char* label;
	int64_t start;
	int64_t end;
};

struct Timer{

	struct Timestamp{
		const char* label;
		int64_t time;
	};

	int64_t t_start;
	int64_t t_end;
	int64_t t_start_nano;
	int64_t t_end_nano;

	Timestamp timestamps[1000];
	int numTimestamps = 0;

	TimerItem measurements[1000];
	int numMeasurements = 0;

	static Timer* instance;

	static void timestamp(const char* label);
	static void print();

	static TimerItem start(const char* label);
	static void stop(TimerItem item);

};


float strlen(const char* string);
float clamp(float value, float min, float max);
bool isFirstThread();
void clearBuffer(int* buffer, int offset, int count, int value);

void drawPoint(Points* points, vec3 pos, uint32_t color);
void drawLine(Lines* lines, vec3 start, vec3 end, uint32_t color);
void drawTriangle(Triangles* triangles, vec3 p1, vec3 p2, vec3 p3, uint32_t color);
void drawBoundingBox(Lines* lines, vec3 pos, vec3 size, uint32_t color);
void drawBox(Triangles* triangles, vec3 pos, vec3 size, uint32_t color);
void syncGlobalAuto(uint32_t* semaphores, Triangles* triangles);
void syncGlobal(uint32_t* semaphores, Triangles* triangles, int semaphoreIndex);
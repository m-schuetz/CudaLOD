
#version 450

#extension GL_NV_gpu_shader5 : enable

layout(location = 0) uniform mat4 uViewProj;

struct Point{
	float x;
	float y;
	float z;
	uint color;
};

layout(std430, binding = 0) buffer data_1 {
	Point points[];
};

out vec3 vColor;

void main() {

	Point point = points[gl_VertexID];

	vec3 pos = vec3(point.x, point.y, point.z);

	gl_Position = vec4(uViewProj * vec4(pos, 1.0));

	vColor = vec3(
		float((point.color >>  0) & 0xFF) / 255.0,
		float((point.color >>  8) & 0xFF) / 255.0,
		float((point.color >> 16) & 0xFF) / 255.0
	);


	// uint32_t faceID = gl_VertexID / 6;
	// uint32_t color = faceID * 12345678;

	// vColor = vec3(
	// 	float((color >>  0) & 0xFF) / 255.0,
	// 	float((color >>  8) & 0xFF) / 255.0,
	// 	float((color >> 16) & 0xFF) / 255.0
	// );
}
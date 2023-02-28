
#version 450

#extension GL_NV_gpu_shader5 : enable

layout(location = 0) uniform mat4 uViewProj;

struct Point{
	float x;
	float y;
	float z;
	uint color;
};

struct Voxel{
	float x;
	float y;
	float z;
	uint color;
	float size;
};

layout(std430, binding = 0) buffer data_1 {
	Voxel voxels[];
};

out vec3 vColor;

void main() {

	int verticesPerBox = 6 * 2 * 3;
	int voxelID = gl_VertexID / verticesPerBox;
	Voxel voxel = voxels[voxelID];

	vec3 pos = vec3(voxel.x, voxel.y, voxel.z);

	gl_Position = vec4(uViewProj * vec4(pos, 1.0));

	gl_PointSize = 1.0;

	vColor = vec3(
		float((voxel.color >>  0) & 0xFF) / 255.0,
		float((voxel.color >>  8) & 0xFF) / 255.0,
		float((voxel.color >> 16) & 0xFF) / 255.0
	);

	// if((gl_VertexID % verticesPerBox) != 0){
	// 	gl_Position = vec4(10.0, 10.0, 10.0, 1.0);
	// }



	vec3 voxelVertices[36] = {
		// BOTTOM
		vec3(-1.0, -1.0, -1.0),
		vec3( 1.0, -1.0, -1.0),
		vec3( 1.0,  1.0, -1.0),
		vec3(-1.0, -1.0, -1.0),
		vec3( 1.0,  1.0, -1.0),
		vec3(-1.0,  1.0, -1.0),

		// TOP
		vec3(-1.0, -1.0,  1.0),
		vec3( 1.0, -1.0,  1.0),
		vec3( 1.0,  1.0,  1.0),
		vec3(-1.0, -1.0,  1.0),
		vec3( 1.0,  1.0,  1.0),
		vec3(-1.0,  1.0,  1.0),

		// FRONT
		vec3(-1.0, -1.0, -1.0 ),
		vec3(-1.0,  1.0, -1.0 ),
		vec3(-1.0,  1.0,  1.0 ),
		vec3(-1.0, -1.0, -1.0 ),
		vec3(-1.0,  1.0,  1.0 ),
		vec3(-1.0, -1.0,  1.0 ),

		// BACK
		vec3( 1.0, -1.0, -1.0 ),
		vec3( 1.0,  1.0, -1.0 ),
		vec3( 1.0,  1.0,  1.0 ),
		vec3( 1.0, -1.0, -1.0 ),
		vec3( 1.0,  1.0,  1.0 ),
		vec3( 1.0, -1.0,  1.0 ),

		// LEFT
		vec3(-1.0, -1.0, -1.0),
		vec3( 1.0, -1.0, -1.0),
		vec3( 1.0, -1.0,  1.0),
		vec3(-1.0, -1.0, -1.0),
		vec3( 1.0, -1.0,  1.0),
		vec3(-1.0, -1.0,  1.0),

		// RIGHT
		vec3(-1.0,  1.0, -1.0),
		vec3( 1.0,  1.0, -1.0),
		vec3( 1.0,  1.0,  1.0),
		vec3(-1.0,  1.0, -1.0),
		vec3( 1.0,  1.0,  1.0),
		vec3(-1.0,  1.0,  1.0),
	};

	int localID = gl_VertexID % verticesPerBox;
	vec3 vertex = 0.5 * voxel.size * voxelVertices[localID] + pos;

	gl_Position = vec4(uViewProj * vec4(vertex, 1.0));
	// gl_Position.x = 10.0;
	// gl_Position.w = 1.0;
	




	// vColor = vec3(1.0, 0.0, 0.0);

	// uint32_t faceID = gl_VertexID / 6;
	// uint32_t color = faceID * 12345678;

	// vColor = vec3(
	// 	float((color >>  0) & 0xFF) / 255.0,
	// 	float((color >>  8) & 0xFF) / 255.0,
	// 	float((color >> 16) & 0xFF) / 255.0
	// );
}
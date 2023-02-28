
#version 450

layout(location = 0) uniform mat4 uViewProj;

struct Vertex{
	float x;
	float y;
	float z;
	uint color;
};

layout(std430, binding = 0) buffer data_1 {
	Vertex vertices[];
};

out vec3 vColor;

void main() {
	Vertex vertex = vertices[gl_VertexID];
	// Vertex vertex = vertices[2 * gl_VertexID + 0];

	vec3 pos = vec3(vertex.x, vertex.y, vertex.z);

	gl_Position = vec4(uViewProj * vec4(pos, 1.0));

	vColor = vec3(
		float((vertex.color >>  0) & 0xFF) / 255.0,
		float((vertex.color >>  8) & 0xFF) / 255.0,
		float((vertex.color >> 16) & 0xFF) / 255.0
	);
}
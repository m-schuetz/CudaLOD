
#version 450

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
	// gl_Position = vec4(0.0, 0.0, 0.5, 1.0);

	// gl_Position = vec4(
	// 	//float(gl_VertexID) / 100.0,
	// 	0.0,
	// 	0.5, 0.5, 1.0
	// );


	// vec2 uv = 0.5 * (gl_Position.xy / gl_Position.w) + 0.5;
	// vec2 uv = (gl_Position.xy / gl_Position.w);

	// gl_PointSize = 2400.0 / gl_Position.w;
	// gl_PointSize = 10.0 / gl_Position.w; // + 10 * length(uv);

	uint size = (point.color >> 24) & 0xFF;
	gl_PointSize = float(size) * 0.5;
	// gl_PointSize = 2.0;

	vColor = vec3(
		float((point.color >>  0) & 0xFF) / 255.0,
		float((point.color >>  8) & 0xFF) / 255.0,
		float((point.color >> 16) & 0xFF) / 255.0
	);

	// if(point.color > 255){
	// 	gl_Position.w = 0.0001;
	// }

	// if(gl_VertexID > 100 * 1000){
	// 	gl_Position.w = 0.0001;
	// }

	// vColor = vec3(1.0, 0.0, 0.0);

	// vColor = vec3(uv, 0.0);

}
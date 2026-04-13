#version 410 core

// ---------------------------------------------------------------------------
// image.vert
//
// Renders a 2D image quad.
//
// VAO layout:
//   location 0 — vec3 position
//   location 1 — vec2 texCoord
//
// ---------------------------------------------------------------------------

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoord;

uniform mat4 u_transform;
uniform mat4 u_projection;
out highp vec2 fragTexCoord;

void main() {
    gl_Position = u_projection * u_transform * vec4(position, 1.0);
    fragTexCoord = texCoord;
}
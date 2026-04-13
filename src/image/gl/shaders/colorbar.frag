#version 410 core

in vec2 v_uv;

out vec4 out_color;

uniform sampler2D u_cmap;

void main() {
    out_color = texture(u_cmap, v_uv);
}
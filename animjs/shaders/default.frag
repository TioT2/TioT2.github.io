#version 300 es

precision highp float;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormalShade;
layout(location = 2) out vec4 outDiffuse;

in vec3 drawPosition;
in vec2 drawTexCoord;
in vec3 drawNormal;

uniform sampler2D Texture0;

void main() {
  outPosition = vec4(drawPosition, 1);
  outNormalShade = vec4(drawNormal, 1);
  outDiffuse = texture(Texture0, drawTexCoord * vec2(1, -1));
}
#version 300 es

precision highp float;

layout(location = 0) out vec4 outPositionShade;
layout(location = 1) out vec4 outNormalRoughness;
layout(location = 2) out vec4 outBasecolorMetallic;

in vec3 drawPosition;
in vec2 drawTexCoord;
in vec3 drawNormal;

uniform sampler2D Texture0;

void main() {
  outPositionShade = vec4(drawPosition, 1.0);
  outNormalRoughness = vec4(normalize(drawNormal), 0.3);
  outBasecolorMetallic = vec4(texture(Texture0, drawTexCoord * vec2(1, -1)).rgb, 0.4);
}
#version 300 es

precision highp float;

layout(location = 0) out vec4 outPositionShade;
layout(location = 1) out vec4 outNormalRoughness;
layout(location = 2) out vec4 outBasecolorMetallic;

void main() {
  outPositionShade = vec4(0.0);
  outNormalRoughness = vec4(0.0);
  outBasecolorMetallic = vec4(0, 1, 0, 1);
} /* main */
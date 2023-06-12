#version 300 es

precision highp float;

layout(location = 0) out vec4 outPositionShade;
layout(location = 1) out vec4 outNormalRoughness;
layout(location = 2) out vec4 outBasecolorMetallic;

in vec3 drawPosition;
in vec2 drawTexCoord;
in vec3 drawNormal;

uniform materialUBO {
  vec3 baseColor;
  float metallic;
  float roughness;
};

void main() {
  outPositionShade = vec4(drawPosition, 1);
  outNormalRoughness = vec4(normalize(drawNormal), roughness);
  outBasecolorMetallic = vec4(baseColor, metallic);
} /* main */

/* default_pbr.frag */
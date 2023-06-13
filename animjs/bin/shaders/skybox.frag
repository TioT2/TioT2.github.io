#version 300 es

precision highp float;

layout(location = 0) out vec4 outPositionShade;
layout(location = 1) out vec4 outNormalRoughness;
layout(location = 2) out vec4 outBasecolorMetallic;

in vec2 drawTexCoord;

uniform samplerCube Texture0;

uniform cameraBuffer {
  mat4 transformWorld;
  mat4 transformViewProj;
};

uniform projectionInfo {
  vec4 camDir;
  vec4 camRight;
  vec4 camUp;
};

void main() {
  outPositionShade = vec4(0);
  outNormalRoughness = vec4(0);

  vec3 dir = normalize(camDir.xyz + camRight.xyz * drawTexCoord.x + camUp.xyz * drawTexCoord.y);
  
  outBasecolorMetallic = vec4(texture(Texture0, dir).xyz, 0);
} /* main */

/* default_pbr.frag */
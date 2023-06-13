#version 300 es

precision highp float;

layout(location = 0) out vec4 outPositionShade;
layout(location = 1) out vec4 outNormalRoughness;
layout(location = 2) out vec4 outBasecolorMetallic;

uniform projectionInfo {
  vec4 camDir;
  vec4 camRight;
  vec4 camUp;
};

uniform sampler2D Texture0;

in vec2 drawTexCoord;
in vec2 inAngles;

#define PI 3.1415926535

void main() {
  outPositionShade = vec4(0);
  outNormalRoughness = vec4(0);

  vec3 dir = camDir.xyz + camRight.xyz * drawTexCoord.x + camUp.xyz * drawTexCoord.y;

  float azimuth = sign(dir.z) * acos(dir.x / length(dir.xz));
  float elevator = acos(dir.y / length(dir));

  outBasecolorMetallic = vec4(texture(Texture0, vec2((azimuth / PI + 1.0) / 2.0, elevator / PI)).xyz, 0);
} /* main */

/* default_pbr.frag */
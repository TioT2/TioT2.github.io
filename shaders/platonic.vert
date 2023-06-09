#version 300 es

precision highp float;

in vec3 inPosition;
in vec2 inTexCoord;
in vec3 inNormal;

uniform cameraBuffer
{
  mat4 transformWorld;
  mat4 transformViewProj;
};

void main()
{
  gl_Position = (transformViewProj * transformWorld) * vec4(inPosition, 1);
} /* main */
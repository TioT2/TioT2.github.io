#version 300 es

precision highp float;

in vec3 inPosition;
in vec2 inTexCoord;
in vec3 inNormal;

out vec3 drawPosition;
out vec2 drawTexCoord;
out vec3 drawNormal;

uniform cameraBuffer
{
  mat4 transformWorld;
  mat4 transformViewProj;
  vec4 cameraLocation;
};

void main()
{
  gl_Position = (transformViewProj * transformWorld) * vec4(inPosition, 1);

  drawPosition = (transformWorld * vec4(inPosition, 1)).xyz;
  drawTexCoord = inTexCoord;
  drawNormal = mat3(inverse(transpose(transformWorld))) * inNormal;
}
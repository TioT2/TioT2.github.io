#version 300 es

precision highp float;

in vec3 inPosition;
in vec2 inTexCoord;
in vec3 inNormal;

out vec2 drawTexCoord;

void main() {
  gl_Position = vec4(inPosition, 1.0);

  drawTexCoord = inTexCoord * 2.0 - vec2(1.0);
} /* main */
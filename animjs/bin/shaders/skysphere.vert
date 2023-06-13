#version 300 es

precision highp float;

in vec2 inTexCoord;

out vec2 drawTexCoord;

void main() {
  drawTexCoord = inTexCoord * 2.0 - vec2(1.0);
  gl_Position = vec4(drawTexCoord, 0.0, 1.0);
} /* main */
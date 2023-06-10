#version 300 es

in vec3 inPosition;
in vec2 inTexCoord;

out vec2 drawTexCoord;

void main() {
  gl_Position = vec4(inPosition, 1);
  drawTexCoord = inTexCoord;
} /* main */

/* target.vert */
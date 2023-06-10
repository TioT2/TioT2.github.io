#version 300 es

precision highp float;

/**/

in vec4 inPos;
out vec2 fragCoord;

void main() {
  gl_Position = inPos;
  fragCoord = inPos.xy;
}

#version 300 es
/**/

in highp vec4 inPos;
out highp vec2 fragCoord;

void main() {
  gl_Position = inPos;
  fragCoord = inPos.xy;
}

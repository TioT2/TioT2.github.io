#version 300 es

precision highp float;

/**/
out vec4 outColor;
in vec2 fragCoord;

uniform vec2 CParam;
uniform vec2 Position;
uniform float Scale;
uniform vec3 Color;
uniform vec2 Projection;

vec2 complMul(vec2 a, vec2 b) {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

float Julia(vec2 z, vec2 c)
{
  float n = 0.0;
  
  while (n < 256.0 && length(z) < 2.0)
    z = complMul(z, z) + c, n++;
  
  return n / 256.0;
}

void main() {
  vec2 Coord = (fragCoord * 2.0 / Scale + Position) * Projection;
  float J = Julia(Coord, CParam);

  outColor = vec4(Color * J, 1);
}
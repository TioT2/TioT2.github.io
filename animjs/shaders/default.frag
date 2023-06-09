#version 300 es

precision highp float;

out vec4 outColor;

in vec3 drawPosition;
in vec2 drawTexCoord;
in vec3 drawNormal;

uniform sampler2D diffuseTexture;

void main() {
  vec3 L = normalize(vec3(1));
  vec3 diffuseColor = texture(diffuseTexture, drawTexCoord * vec2(1, -1)).xyz;

  outColor = vec4(diffuseColor * clamp(dot(drawNormal, L), 0.1, 1.0), 1);
}
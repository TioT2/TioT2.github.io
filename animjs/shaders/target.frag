#version 300 es

precision highp float;

layout(location = 0) out vec4 outColor;

in vec2 drawTexCoord;

uniform sampler2D Texture0;
uniform sampler2D Texture1;
uniform sampler2D Texture2;

void main() {
  vec4 position = texture(Texture0, drawTexCoord);
  vec4 normalShade = texture(Texture1, drawTexCoord);
  vec4 diffuse = texture(Texture2, drawTexCoord);
  vec3 lightDirection = normalize(vec3(1));

  if (normalShade.w == 1.0) {
    outColor = vec4((diffuse * dot(normalShade.xyz, lightDirection)).xyz, 1);
  } else {
    outColor = vec4(diffuse.xyz, 1);
  }
} /* main */

/* target.frag */
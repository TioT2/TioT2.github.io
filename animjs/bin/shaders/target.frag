#version 300 es

precision highp float;

layout(location = 0) out vec4 outColor;

in vec2 drawTexCoord;

float GGX_PartialGeometry(float cosThetaN, float alpha) {
  float cosTheta_sqr = cosThetaN * cosThetaN;
  float tan2 = (1.0 - cosTheta_sqr) / cosTheta_sqr;

  return 2.0 / (1.0 + sqrt(1.0 + alpha * alpha * tan2));
} /* GGX_PartialGeometry */

float PI = 3.14159265357989;

float GGX_Distribution(float cosThetaNH, float alpha) {
  float alphaSqr = alpha * alpha;
  float NHSqr = clamp(cosThetaNH * cosThetaNH, 0.0, 1.0);
  float den = NHSqr * alphaSqr + (1.0 - NHSqr);

  return alphaSqr / (PI * den * den);
} /* GGX_Distribution */

vec3 FrenelSchlick(vec3 F0, float cosTheta) {
  return F0 + (vec3(1.0) - F0) * pow(1.0 - clamp(cosTheta, 0.0, 1.0), 5.0);
} /* FrenelSchlick */

vec3 shade(vec3 lightDirection, vec3 normal, vec3 viewDirection, float roughness, vec3 F0, vec3 albedo) {
  float nl = dot(normal, lightDirection);
  if (nl <= 0.0)
    return vec3(0.0);
  float nv = dot(normal, viewDirection);
  if (nv <= 0.0)
    return vec3(0.0);
  vec3 h = normalize(viewDirection + lightDirection);
  float nh = dot(normal, h);
  float hv = dot(viewDirection, h);

  float roughnessSqr = roughness * roughness;
  float G = GGX_PartialGeometry(nv, roughnessSqr) * GGX_PartialGeometry(nl, roughnessSqr);
  float D = GGX_Distribution(nh, roughnessSqr);
  vec3 F = FrenelSchlick(F0, hv);

  vec3 specK = F * D * G * 0.25 / hv;
  vec3 diffK = albedo * clamp(vec3(1.0) - F, vec3(0.0), vec3(1.0)) * nl / PI;
  return max(vec3(0), diffK + specK);
} /* shade */

uniform sampler2D Texture0;
uniform sampler2D Texture1;
uniform sampler2D Texture2;

uniform cameraBuffer
{
  mat4 transformWorld;
  mat4 transformViewProj;
  vec4 cameraLocation;
};

void main() {
  vec4 positionShade = texture(Texture0, drawTexCoord);
  vec4 basecolorMetallic = texture(Texture2, drawTexCoord);

  if (positionShade.w == 1.0) {
    vec4 normalRoughness = texture(Texture1, drawTexCoord);
    vec3 lightDirection = normalize(vec3(1));

    vec3 F0 = mix(vec3(0), basecolorMetallic.rgb, vec3(basecolorMetallic.a));
    vec3 albedo = mix(basecolorMetallic.rgb, vec3(0.04), vec3(basecolorMetallic.a));

    outColor = vec4(shade(lightDirection, normalRoughness.xyz, normalize(cameraLocation.xyz - positionShade.xyz), normalRoughness.w, F0, albedo), 1);

    outColor.xyz = outColor.xyz / (outColor.xyz + vec3(1));
    outColor.xyz = pow(outColor.xyz, vec3(1.0 / 2.2));
  } else {
    outColor = vec4(basecolorMetallic.rgb, 1);
  }
} /* main */

/* target.frag */
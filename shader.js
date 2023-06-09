async function loadShaderModule(gl, type, path) {
  const src = await fetch(path).then(response => response.text());
  const shader = gl.createShader(type);

  gl.shaderSource(shader, src);
  gl.compileShader(shader);

  const res = gl.getShaderInfoLog(shader);
  if (res != null && res.length > 0)
    console.error(`Shader module compilation error: ${res}`);

  return shader;
} /* loadShaderModule */

export async function loadShader(gl, path) {

  // Wait compilation of all shaders
  let res = await Promise.all([
    loadShaderModule(gl, gl.VERTEX_SHADER, path + ".vert?" + Math.random().toString()),
    loadShaderModule(gl, gl.FRAGMENT_SHADER, path + ".frag?" + Math.random().toString()),
  ]);

  let program = gl.createProgram();

  gl.attachShader(program, res[0]);
  gl.attachShader(program, res[1]);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS))
    console.error(`Shader ${path} linking error: ${gl.getProgramInfoLog(program)}`);

  return program;
} /* loadShader */
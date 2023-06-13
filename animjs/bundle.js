function shaderModuleFromSource(gl, type, source) {
  if (source == null) {
    return null;
  }

  let shader = gl.createShader(type);

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  let res = gl.getShaderInfoLog(shader);
  if (res != null && res.length > 0)
    console.error(`Shader module compilation error: ${res}`);

  return shader;
} /* shaderModuleFromSource */

function shaderFromSource(gl, vertSource, geomSource, fragSource) {
  let modules = [
    shaderModuleFromSource(gl, gl.VERTEX_SHADER, vertSource),
    shaderModuleFromSource(gl, gl.GEOMETRY_SHADER, geomSource),
    shaderModuleFromSource(gl, gl.FRAGMENT_SHADER, fragSource),
  ];
  let program = gl.createProgram();

  for (let i = 0; i < 3; i++) {
    if (modules[i] != null) {
      gl.attachShader(program, modules[i]);
    }
  }

  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS))
    console.error(`Shader linking error: ${gl.getProgramInfoLog(program)}`);

  return program;
} /* shaderFromSource */

async function loadShader(gl, path) {
  return shaderFromSource(gl,
    await fetch(path + ".vert?" + Math.random().toString()).then(response => response.ok ? response.text() : null),
    await fetch(path + ".geom?" + Math.random().toString()).then(response => response.ok ? response.text() : null),
    await fetch(path + ".frag?" + Math.random().toString()).then(response => response.ok ? response.text() : null),
  );
} /* loadShader */

class Vec3 {
  x;
  y;
  z;

  constructor(nx, ny, nz) {
    this.x = nx;
    this.y = ny;
    this.z = nz;
  } /* constructor */

  copy() {
    return new Vec3(this.x, this.y, this.z);
  } /* copy */

  add(m2) {
    return new Vec3(this.x + m2.x, this.y + m2.y, this.z + m2.z);
  } /* add */

  sub(m2) {
    return new Vec3(this.x - m2.x, this.y - m2.y, this.z - m2.z);
  } /* sub */

  mul(m2) {
    if (typeof(m2) == "Vec3")
      return new Vec3(this.x * m2.x, this.y * m2.y, this.z * m2.z);
    else
      return new Vec3(this.x * m2,   this.y * m2,   this.z * m2  );
  } /* mul */

  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  } /* length */
  
  dot(m2) {
    return this.x * m2.x + this.y * m2.y + this.z * m2.z;
  } /* dot */

  cross(othv) {
    return new Vec3(
      this.y * othv.z - othv.y * this.z,
      this.z * othv.x - othv.z * this.x,
      this.x * othv.y - othv.x * this.y
    );
  } /* cross */

  normalize() {
    let len = this.length();

    return new Vec3(this.x / len, this.y / len, this.z / len);
  } /* normalize */

  static fromSpherical(azimuth, elevation, radius = 1) {
    return new Vec3(
      radius * Math.sin(elevation) * Math.cos(azimuth),
      radius * Math.cos(elevation),
      radius * Math.sin(elevation) * Math.sin(azimuth)
    );
  } /* sphericalToCartesian */
} /* Vec3 */

class Vec2 {
  x;
  y;

  constructor(nx, ny) {
    this.x = nx;
    this.y = ny;
  } /* constructor */

  copy() {
    return new Vec2();
  }

  add(m2) {
    return new Vec2(this.x + m2.x, this.y + m2.y);
  } /* add */

  sub(m2) {
    return new Vec2(this.x - m2.x, this.y - m2.y);
  } /* sub */

  mul(m2) {
    if (typeof(m2) == "Vec2")
      return new Vec2(this.x * m2.x, this.y * m2.y);
    else
      return new Vec2(this.x * m2,   this.y * m2);
  } /* mul */

  length2() {
    return this.x * this.x + this.y * this;  } /* length2 */

  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y);
  } /* length */

  dot(m2) {
    return this.x * m2.x + this.y * m2.y;
  } /* dot */

  cross(othv) {
    return this.x * othv.y - othv.x * this.y;
  } /* cross */

  normalize() {
    let len = this.length();

    return new Vec2(this.x / len, this.y / len);
  } /* normalize */
} /* Vec2 */

class Size {
  w;
  h;

  constructor(w, h) {
    this.w = w;
    this.h = h;
  } /* constructor */

  copy() {
    return new Size(this.w, this.h);
  } /* copy */
} /* Size */

class Mat4 {
  m;

  constructor(v00, v01, v02, v03,
              v10, v11, v12, v13,
              v20, v21, v22, v23,
              v30, v31, v32, v33) {
    this.m = [
      v00, v01, v02, v03,
      v10, v11, v12, v13,
      v20, v21, v22, v23,
      v30, v31, v32, v33
    ];
  } /* constructor */

  copy() {
    return new Mat4(
      this.m[ 0], this.m[ 1], this.m[ 2], this.m[ 3],
      this.m[ 4], this.m[ 5], this.m[ 6], this.m[ 7],
      this.m[ 8], this.m[ 9], this.m[10], this.m[11],
      this.m[12], this.m[13], this.m[14], this.m[15],
    );
  } /* copy */

  transformPoint(v)
  {
    return new Vec3(
      v.x * this.m[ 0] + v.y * this.m[ 4] + v.z * this.m[ 8] + this.m[12],
      v.x * this.m[ 1] + v.y * this.m[ 5] + v.z * this.m[ 9] + this.m[13],
      v.x * this.m[ 2] + v.y * this.m[ 6] + v.z * this.m[10] + this.m[14]
    );
  } /* transformPoint */

  transform4x4(v)
  {
    let w = V.x * this.m[3] + V.y * this.m[7] + V.z * this.m[11] + this.m[15];
  
    return new Vec3(
      (v.x * this.m[ 0] + v.y * this.m[ 4] + v.z * this.m[ 8] + this.m[12]) / w,
      (v.x * this.m[ 1] + v.y * this.m[ 5] + v.z * this.m[ 9] + this.m[13]) / w,
      (v.x * this.m[ 2] + v.y * this.m[ 6] + v.z * this.m[10] + this.m[14]) / w
    );
  } /* transform4x4 */

  transpose() {
    return new Mat4(
      this.m[ 0], this.m[ 4], this.m[ 8], this.m[12],
      this.m[ 1], this.m[ 5], this.m[ 9], this.m[13],
      this.m[ 2], this.m[ 6], this.m[10], this.m[14],
      this.m[ 3], this.m[ 7], this.m[11], this.m[15]
    );
  } /* transpose */

  mul(m2) {
    return new Mat4(
      this.m[ 0] * m2.m[ 0] + this.m[ 1] * m2.m[ 4] + this.m[ 2] * m2.m[ 8] + this.m[ 3] * m2.m[12],
      this.m[ 0] * m2.m[ 1] + this.m[ 1] * m2.m[ 5] + this.m[ 2] * m2.m[ 9] + this.m[ 3] * m2.m[13],
      this.m[ 0] * m2.m[ 2] + this.m[ 1] * m2.m[ 6] + this.m[ 2] * m2.m[10] + this.m[ 3] * m2.m[14],
      this.m[ 0] * m2.m[ 3] + this.m[ 1] * m2.m[ 7] + this.m[ 2] * m2.m[11] + this.m[ 3] * m2.m[15],

      this.m[ 4] * m2.m[ 0] + this.m[ 5] * m2.m[ 4] + this.m[ 6] * m2.m[ 8] + this.m[ 7] * m2.m[12],
      this.m[ 4] * m2.m[ 1] + this.m[ 5] * m2.m[ 5] + this.m[ 6] * m2.m[ 9] + this.m[ 7] * m2.m[13],
      this.m[ 4] * m2.m[ 2] + this.m[ 5] * m2.m[ 6] + this.m[ 6] * m2.m[10] + this.m[ 7] * m2.m[14],
      this.m[ 4] * m2.m[ 3] + this.m[ 5] * m2.m[ 7] + this.m[ 6] * m2.m[11] + this.m[ 7] * m2.m[15],

      this.m[ 8] * m2.m[ 0] + this.m[ 9] * m2.m[ 4] + this.m[10] * m2.m[ 8] + this.m[11] * m2.m[12],
      this.m[ 8] * m2.m[ 1] + this.m[ 9] * m2.m[ 5] + this.m[10] * m2.m[ 9] + this.m[11] * m2.m[13],
      this.m[ 8] * m2.m[ 2] + this.m[ 9] * m2.m[ 6] + this.m[10] * m2.m[10] + this.m[11] * m2.m[14],
      this.m[ 8] * m2.m[ 3] + this.m[ 9] * m2.m[ 7] + this.m[10] * m2.m[11] + this.m[11] * m2.m[15],

      this.m[12] * m2.m[ 0] + this.m[13] * m2.m[ 4] + this.m[14] * m2.m[ 8] + this.m[15] * m2.m[12],
      this.m[12] * m2.m[ 1] + this.m[13] * m2.m[ 5] + this.m[14] * m2.m[ 9] + this.m[15] * m2.m[13],
      this.m[12] * m2.m[ 2] + this.m[13] * m2.m[ 6] + this.m[14] * m2.m[10] + this.m[15] * m2.m[14],
      this.m[12] * m2.m[ 3] + this.m[13] * m2.m[ 7] + this.m[14] * m2.m[11] + this.m[15] * m2.m[15],
    );
  } /* mul */

  static identity() {
    return new Mat4(
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    );
  } /* identity */

  static scale(s) {
    return new Mat4(
      s.x, 0,   0,   0,
      0,   s.y, 0,   0,
      0,   0,   s.z, 0,
      0,   0,   0,   1
    );
  } /* scale */

  static translate(t) {
    return new Mat4(
      1,   0,   0,   0,
      0,   1,   0,   0,
      0,   0,   1,   0,
      t.x, t.y, t.z, 1
    );
  } /* translate */

  static rotateX(angle) {
    let s = Math.sin(angle), c = Math.cos(angle);

    return new Mat4(
      1, 0, 0, 0,
      0, c, s, 0,
      0,-s, c, 0,
      0, 0, 0, 1
    );
  } /* rotateX */

  static rotateY(angle) {
    let s = Math.sin(angle), c = Math.cos(angle);

    return new Mat4(
      c, 0,-s, 0,
      0, 1, 0, 0,
      s, 0, c, 0,
      0, 0, 0, 1
    );
  } /* rotateY */

  static rotateZ(angle) {
    let s = Math.sin(angle), c = Math.cos(angle);

    return new Mat4(
      c, s, 0, 0,
     -s, c, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    );
  } /* rotateZ */

  static view(loc, at, up) {
    let
      dir = at.sub(loc).normalize(),
      rgh = dir.cross(up).normalize(),
      tup = rgh.cross(dir);

    return new Mat4(
      rgh.x,         tup.x,         -dir.x,       0,
      rgh.y,         tup.y,         -dir.y,       0,
      rgh.z,         tup.z,         -dir.z,       0,
      -loc.dot(rgh), -loc.dot(tup), loc.dot(dir), 1
    );
  } /* view */

  static frustum(left, right, bottom, top, near, far) {
    return new Mat4(
      2 * near / (right - left),       0,                               0,                              0,
      0,                               2 * near / (top - bottom),       0,                              0,
      (right + left) / (right - left), (top + bottom) / (top - bottom), (near + far) / (near - far),   -1,
      0,                               0,                               2 * near * far / (near - far),  0
    );
  } /* frustum */
} /* Mat4 */

class Camera {
  // camera projection shape params
  projSize = new Size(0.01, 0.01);
  near = 0.01;
  far = 8192;
  
  // current screen resolution
  screenSize;

  // camera location
  loc;
  at;
  dir;
  up;
  right;

  // camera projection matrices
  view;
  proj;
  viewProj;

  constructor() {
    this.proj = Mat4.identity();
    this.set(new Vec3(0, 0, -1), new Vec3(0, 0, 0), new Vec3(0, 1, 0));
    this.resize(new Size(30, 30));
  } /* constructor */

  projSet(newNear, newFar, newProjSize = new Size(0.01, 0.01)) {
    this.projSize = newProjSize.copy();
    this.near = newNear;
    this.far = newFar;

    if (this.screenSize.w > this.screenSize.h) {
      this.projSize.w *= this.screenSize.w / this.screenSize.h;
    } else {
      this.projSize.h *= this.screenSize.h / this.screenSize.w;
    }

    this.proj = Mat4.frustum(-this.projSize.w / 2, this.projSize.w / 2, -this.projSize.h / 2, this.projSize.h / 2, this.near, this.far);
    this.viewProj = this.view.mul(this.proj);
  } /* projSet */

  resize(newScreenSize) {
    this.screenSize = newScreenSize.copy();
    this.projSet(this.near, this.far);
  } /* resize */

  set(loc, at, up) {
    this.view = Mat4.view(loc, at, up);
    this.viewProj = this.view.mul(this.proj);

    this.loc = loc.copy();
    this.at = at.copy();

    this.right = new Vec3(this.view.m[ 0], this.view.m[ 4], this.view.m[ 8]);
    this.up    = new Vec3(this.view.m[ 1], this.view.m[ 5], this.view.m[ 9]);
    this.dir   = new Vec3(this.view.m[ 2], this.view.m[ 6], this.view.m[10]).mul(-1);
  } /* set */
} /* Camera */

/* format decoding function */
function getFormat(componentType, componentCount) {
  const fmts = [WebGL2RenderingContext.RED, WebGL2RenderingContext.RG, WebGL2RenderingContext.RGB, WebGL2RenderingContext.RGBA];
  switch (componentType) {
    case Texture.FLOAT:
      const floatInternals = [WebGL2RenderingContext.R32F, WebGL2RenderingContext.RG32F, WebGL2RenderingContext.RGB32F, WebGL2RenderingContext.RGBA32F];
      return {
        format: fmts[componentCount - 1],
        internal: floatInternals[componentCount - 1],
        componentType: WebGL2RenderingContext.FLOAT,
      };

    case Texture.UNSIGNED_BYTE:
      const byteInternals = [WebGL2RenderingContext.R8, WebGL2RenderingContext.RG8, WebGL2RenderingContext.RGB8, WebGL2RenderingContext.RGBA8];
      return {
        format: fmts[componentCount - 1],
        internal: byteInternals[componentCount - 1],
        componentType: WebGL2RenderingContext.UNSIGNED_BYTE,
      };

    case Texture.DEPTH:
      return {
        format: WebGL2RenderingContext.DEPTH_COMPONENT,
        internal: WebGL2RenderingContext.DEPTH_COMPONENT32F,
        componentType: WebGL2RenderingContext.FLOAT,
      };

    default:
      // minimal format possible
      return {
        format: WebGL2RenderingContext.RED,
        internal: WebGL2RenderingContext.R8,
        componentType: WebGL2RenderingContext.UNSIGNED_BYTE,
      };
  }
} /* getFormat */

class Texture {
  #gl;
  #format;
  size;
  id;

  static FLOAT         = 0;
  static UNSIGNED_BYTE = 1;
  static DEPTH         = 2;

  constructor(gl, componentType = Texture.FLOAT, componentCount = 1) {
    this.gl = gl;
    this.size = new Size(1, 1);
    this.id = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.id);

    this.format = getFormat(componentType, componentCount);
    // put empty image data
    gl.texImage2D(gl.TEXTURE_2D, 0, this.format.internal, 1, 1, 0, this.format.format, this.format.componentType, null);

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  } /* constructor */

  
  static defaultCheckerTexture = null;
  static defaultChecker(gl) {
    if (Texture.defaultCheckerTexture === null) {
      Texture.defaultCheckerTexture = new Texture(gl, Texture.UNSIGNED_BYTE, 4);
  
      gl.bindTexture(gl.TEXTURE_2D, Texture.defaultCheckerTexture.id);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0x00, 0xFF, 0x00, 0xFF]));

      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    }

    return Texture.defaultCheckerTexture;
  } /* defaultChecker */

  defaultChecker(data) {
    this.gl.bindTexture(gl.TEXTURE_2D, this.id);
    this.gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 2, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([
      0x00, 0xFF, 0x00, 0xFF,
      0x00, 0x00, 0x00, 0xFF,
      0x00, 0x00, 0x00, 0xFF,
      0x00, 0xFF, 0x00, 0xFF,
    ]));
  }

  load(path) {
    let gl = this.gl;

    // async image loading
    let image = new Image();
    image.src = path;
    image.addEventListener("load", () => {
      this.size.w = image.width;
      this.size.h = image.height;
      gl.bindTexture(gl.TEXTURE_2D, this.id);
      gl.texImage2D(gl.TEXTURE_2D, 0, this.format.internal, this.format.format, this.format.componentType, image);
    });
  } /* load */

  bind(program, index = 0) {
    let gl = this.gl;

    gl.activeTexture(gl.TEXTURE0 + index);
    gl.bindTexture(gl.TEXTURE_2D, this.id);

    let location = gl.getUniformLocation(program, `Texture${index}`);
    gl.uniform1i(location, index);
  } /* bind */

  resize(size) {
    let gl = this.gl;
    this.size = size.copy();
    gl.texImage2D(gl.TEXTURE_2D, 0, this.format.internal, this.size.w, this.size.h, 0, this.format.format, this.format.componentType, null);
  } /* resize */
} /* Texture */

const faceDescriptions = [
  {target: WebGL2RenderingContext.TEXTURE_CUBE_MAP_POSITIVE_X, text: "+X", path: "posX"},
  {target: WebGL2RenderingContext.TEXTURE_CUBE_MAP_NEGATIVE_X, text: "-X", path: "negX"},
  {target: WebGL2RenderingContext.TEXTURE_CUBE_MAP_POSITIVE_Y, text: "+Y", path: "posY"},
  {target: WebGL2RenderingContext.TEXTURE_CUBE_MAP_NEGATIVE_Y, text: "-Y", path: "negY"},
  {target: WebGL2RenderingContext.TEXTURE_CUBE_MAP_POSITIVE_Z, text: "+Z", path: "posZ"},
  {target: WebGL2RenderingContext.TEXTURE_CUBE_MAP_NEGATIVE_Z, text: "-Z", path: "negZ"},
];

class Cubemap {
  #gl;
  id;

  constructor(gl) {
    this.gl = gl;

    let ctx = document.createElement("canvas").getContext("2d");

    ctx.canvas.width = 128;
    ctx.canvas.height = 128;

    function drawFace(text) {
      const {width, height} = ctx.canvas;

      ctx.fillStyle = '#CCC';
      ctx.fillRect(0, 0, width, height);
      ctx.font = `${width * 0.5}px consolas`;
      ctx.textAlign = 'center';
      ctx.textBaseLine = 'middle';
      ctx.fillStyle = '#333';

      ctx.fillText(text, width / 2, height / 2);
    }

    this.id = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, this.id);

    for (let descr of faceDescriptions) {
      drawFace(descr.text);

      gl.texImage2D(descr.target, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, ctx.canvas);
    }
    gl.generateMipmap(gl.TEXTURE_CUBE_MAP);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  
    ctx.canvas.remove();
  } /* constructor */

  load(path) {
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, this.id);

    for (let descr of faceDescriptions) {
      new Image();

      gl.texImage2D(descr.target, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, ctx.canvas);
    }
    gl.generateMipmap(gl.TEXTURE_CUBE_MAP);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  } /* load */

  bind(program, index = 0) {
    let gl = this.gl;

    gl.activeTexture(gl.TEXTURE0 + index);
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, this.id);

    let location = gl.getUniformLocation(program, `Texture${index}`);
    gl.uniform1i(location, index);
  } /* bind */
} /* Cubemap */

class UBO {
  gl;
  buffer;
  isEmpty = true;

  constructor(gl) {
    this.gl = gl;
    this.buffer = gl.createBuffer();
  } /* constructor */

  writeData(dataAsFloatArray) {
    this.isEmpty = false;
    this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.buffer);
    this.gl.bufferData(this.gl.UNIFORM_BUFFER, dataAsFloatArray, this.gl.STATIC_DRAW);
  } /* writeData */

  bind(shader, bindingPoint, bufferName) {
    if (!this.isEmpty) {
      let location = this.gl.getUniformBlockIndex(shader, bufferName);

      if (location != -1) {
        this.gl.uniformBlockBinding(shader, location, bindingPoint);
        this.gl.bindBufferBase(this.gl.UNIFORM_BUFFER, bindingPoint, this.buffer);
      }
    }
  } /* bind */
} /* UBO */

class Material {
  uboNameOnShader = "";
  gl;
  ubo = null;    // object buffer
  textures = []; // array of textures
  shader;        // shader pointer

  constructor(gl, shader) {
    this.gl = gl;
    this.shader = shader;
  } /* constructor */

  apply() {
    let gl = this.gl;

    gl.useProgram(this.shader);

    if (this.ubo != null)
      this.ubo.bind(this.shader, 0, this.uboNameOnShader);
    for (let i = 0; i < this.textures.length; i++)
      this.textures[i].bind(this.shader, i);
  } /* apply */
} /* Material */

class Vertex {
  p;
  t;
  n;

  constructor(position, texcoord, normal) {
    this.p = position;
    this.t = texcoord;
    this.n = normal;
  } /* constructor */

  static fromCoord(px, py, pz, pu = 0, pv = 0, pnx = 0, pny = 0, pnz = 1) {
    return new Vertex(new Vec3(px, py, pz), new Vec2(pu, pv), new Vec3(pnx, pny, pnz));
  } /* fromCoord */

  static fromVectors(p, t = new Vec2(0, 0), n = new Vec3(1, 1, 1)) {
    return new Vertex(p.copy(), t.copy(), n.copy());
  } /* fromVectors */
}
class Topology {
  vtx;
  idx;
  type = Topology.TRIANGLES;

  static LINES          = WebGL2RenderingContext.LINES;
  static LINE_STRIP     = WebGL2RenderingContext.LINE_STRIP;
  static LINE_LOOP      = WebGL2RenderingContext.LINE_LOOP;

  static POINTS         = WebGL2RenderingContext.POINTS;

  static TRIANGLES      = WebGL2RenderingContext.TRIANGLES;
  static TRIANGLE_STRIP = WebGL2RenderingContext.TRIANGLE_STRIP;
  static TRIANGLE_FAN   = WebGL2RenderingContext.TRIANGLE_FAN;

  constructor(nvtx = [], nidx = null) {
    this.vtx = nvtx;
    this.idx = nidx;
  } /* constructor */

  static geometryTypeToGL(geometryType) {
    return geometryType;
  } /* geometryTypeToGL */

  static square() {
    let tpl = new Topology([
      Vertex.fromCoord(-1, -1, 0, 0, 0),
      Vertex.fromCoord(-1,  1, 0, 0, 1),
      Vertex.fromCoord( 1, -1, 0, 1, 0),
      Vertex.fromCoord( 1,  1, 0, 1, 1)
    ], [0, 1, 2, 3]);
    tpl.type = Topology.TRIANGLE_STRIP;
    return tpl;
  } /* theTriangle */

  static #planeIndexed(width = 30, height = 30) {
    let tpl = new Topology();

    tpl.type = Topology.TRIANGLE_STRIP;
    tpl.vtx = [];
    tpl.idx = [];

    let i = 0;
    for (let y = 0; y < height - 1; y++) {
      for (let x = 0; x < width; x++) {
        tpl.idx[i++] = y * width + x;
        tpl.idx[i++] = (y + 1) * width + x;
      }
      tpl.idx[i++] = -1;
    }

    return tpl;
  } /* planeIndexed */

  static sphere(radius = 1, width = 30, height = 30) {
    let tpl = Topology.#planeIndexed(width, height);

    for (let y = 0; y < height; y++) {
      let theta = Math.PI * y / (height - 1);
      let stheta = Math.sin(theta);
      let ctheta = Math.cos(theta);

      for (let x = 0; x < width; x++) {
        let phi = 2 * Math.PI * x / (width - 1);

        let nx = stheta * Math.sin(phi);
        let ny = ctheta;
        let nz = stheta * Math.cos(phi);

        tpl.vtx[y * width + x] = Vertex.fromCoord(
          radius * nx, radius * ny, radius * nz,
          x / (width - 1), y / (height - 1),
          nx, ny, nz
        );
      }
    }

    return tpl;
  } /* sphere */

  static async model_obj(path) {
    let tpl = new Topology();
    tpl.vtx = [];
    tpl.type = Topology.TRIANGLES;

    const src = await fetch(path).then(response => response.text());
    let lines = src.split("\n");
    let positions = [];
    let texCoords = [];
    let normals = [];

    for (let li = 0, lineCount = lines.length; li < lineCount; li++) {
      let segments = lines[li].split(" ");

      switch (segments[0]) {
        case "v":
          positions.push(new Vec3(
            parseFloat(segments[1]),
            parseFloat(segments[2]),
            parseFloat(segments[3])
          ));
          break;

        case "vt":
          texCoords.push(new Vec2(
            parseFloat(segments[1]),
            parseFloat(segments[2])
          ));
          break;

        case "vn":
          normals.push(new Vec3(
            parseFloat(segments[1]),
            parseFloat(segments[2]),
            parseFloat(segments[3])
          ));
          break;

        case "f":
          {
            let vtd = segments[1].split("/");
            let i0 = parseInt(vtd[0]), i1 = parseInt(vtd[1]), i2 = parseInt(vtd[2]);

            tpl.vtx.push(new Vertex(
              Number.isNaN(i0) ? new Vec3(0, 0, 0) : positions[i0 - 1],
              Number.isNaN(i1) ? new Vec2(0, 0) : texCoords[i1 - 1],
              Number.isNaN(i2) ? new Vec3(0, 0, 0) : normals[i2 - 1]
            ));
          }
          {
            let vtd = segments[2].split("/");
            let i0 = parseInt(vtd[0]), i1 = parseInt(vtd[1]), i2 = parseInt(vtd[2]);

            tpl.vtx.push(new Vertex(
              Number.isNaN(i0) ? new Vec3(0, 0, 0) : positions[i0 - 1],
              Number.isNaN(i1) ? new Vec2(0, 0) : texCoords[i1 - 1],
              Number.isNaN(i2) ? new Vec3(0, 0, 0) : normals[i2 - 1]
            ));
          }
          {
            let vtd = segments[3].split("/");
            let i0 = parseInt(vtd[0]), i1 = parseInt(vtd[1]), i2 = parseInt(vtd[2]);

            tpl.vtx.push(new Vertex(
              Number.isNaN(i0) ? new Vec3(0, 0, 0) : positions[i0 - 1],
              Number.isNaN(i1) ? new Vec2(0, 0) : texCoords[i1 - 1],
              Number.isNaN(i2) ? new Vec3(0, 0, 0) : normals[i2 - 1]
            ));
          }
        break;
      }
    }

    return tpl;
  } /* model_obj */

  getVerticesAsFloatArray() {
    let res_array = new Float32Array(this.vtx.length * 8);
    let i = 0;
    let mi = this.vtx.length * 8;

    while(i < mi) {
      let vt = this.vtx[i >> 3];
  
      res_array[i++] = vt.p.x;
      res_array[i++] = vt.p.y;
      res_array[i++] = vt.p.z;

      res_array[i++] = vt.t.x;
      res_array[i++] = vt.t.y;

      res_array[i++] = vt.n.x;
      res_array[i++] = vt.n.y;
      res_array[i++] = vt.n.z;
    }

    return res_array;
  } /* getVerticesAsFloatArray */

  getIndicesAsUintArray() {
    return new Uint32Array(this.idx);
  } /* getIndicesAsUintArray */
} /* Topology */

class Primitive {
  gl;
  vertexArrayObject = null;
  indexBuffer = null;
  vertexBuffer = null;
  vertexNumber = 0;
  indexNumber = 0;
  geometryType = Topology.TRIANGLES;
  material = null;

  constructor(glContext) {
    this.gl = glContext;
  } /* constructor */

  draw(cameraBuffer = null) {
    let gl = this.gl;

    this.material.apply();

    if (cameraBuffer != null) {
      cameraBuffer.bind(this.material.shader, 1, "cameraBuffer");
    }

    gl.bindVertexArray(this.vertexArrayObject);
    if (this.indexBuffer != null) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
      gl.drawElements(Topology.geometryTypeToGL(this.geometryType), this.indexNumber, gl.UNSIGNED_INT, 0);
    } else {
      gl.drawArrays(Topology.geometryTypeToGL(this.geometryType), 0, this.vertexNumber);
    }
  } /* draw */

  static async fromTopology(gl, tpl, material) {
    let prim = new Primitive(gl);
    prim.material = material;

    prim.geometryType = tpl.type;

    // Create vertex array
    prim.vertexArrayObject = gl.createVertexArray();
    gl.bindVertexArray(prim.vertexArrayObject);

    // Write vertex buffer
    prim.vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, prim.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, tpl.getVerticesAsFloatArray(), gl.STATIC_DRAW);
    prim.vertexNumber = tpl.vtx.length;

    // Map vertex layout
    let positionLocation = gl.getAttribLocation(prim.material.shader, "inPosition");
    if (positionLocation != -1) {
      gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 8 * 4, 0);
      gl.enableVertexAttribArray(positionLocation);
    }

    let texCoordLocation = gl.getAttribLocation(prim.material.shader, "inTexCoord");
    if (texCoordLocation != -1) {
      gl.vertexAttribPointer(texCoordLocation, 3, gl.FLOAT, false, 8 * 4, 3 * 4);
      gl.enableVertexAttribArray(texCoordLocation);
    }

    let normalLocation = gl.getAttribLocation(prim.material.shader, "inNormal");
    if (normalLocation != -1) {
      gl.vertexAttribPointer(normalLocation, 3, gl.FLOAT, false, 8 * 4, 5 * 4);
      gl.enableVertexAttribArray(normalLocation);
    }

    // Create index buffer
    if (tpl.idx == null) {
      prim.indexBuffer = null;
      prim.indexNumber = 0;
    } else {
      prim.indexBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, prim.indexBuffer);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, tpl.getIndicesAsUintArray(), gl.STATIC_DRAW);
      prim.indexNumber = tpl.idx.length;
    }

    return prim;
  } /* fromArray */
}

function decodeFramebufferStatus(status) {
  switch (status) {
    case WebGL2RenderingContext.FRAMEBUFFER_COMPLETE:                      return "complete";
    case WebGL2RenderingContext.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:         return "incomplete attachment";
    case WebGL2RenderingContext.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:         return "height and width of attachment aren't same";
    case WebGL2RenderingContext.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: return "attachment missing";
    case WebGL2RenderingContext.FRAMEBUFFER_UNSUPPORTED:                   return "attachment format isn't supported";
  }
}

class Target {
  #gl;
  FBO;
  attachments = [];
  size;
  depth;

  constructor(gl, attachmentCount) {
    this.size = new Size(800, 600);
    this.gl = gl;
    this.FBO = gl.createFramebuffer();

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.FBO);

    // create target textures
    let drawBuffers = [];
    for (let i = 0; i < attachmentCount; i++) {
      this.attachments[i] = new Texture(gl, Texture.FLOAT, 4);
      drawBuffers.push(gl.COLOR_ATTACHMENT0 + i);
    }
    gl.drawBuffers(drawBuffers);

    for (let i = 0; i < attachmentCount; i++) {
      gl.bindTexture(gl.TEXTURE_2D, this.attachments[i].id);
      this.attachments[i].resize(this.size);
  
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, this.attachments[i].id, 0);
    }
    this.depth = new Texture(gl, Texture.DEPTH);
    this.depth.resize(this.size);
    gl.bindTexture(gl.TEXTURE_2D, this.depth.id);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.depth.id, 0);

    console.log(`Framebuffer status: ${decodeFramebufferStatus(gl.checkFramebufferStatus(gl.FRAMEBUFFER))}`);
  } /* constructor */

  resize(size) {
    let gl = this.gl;

    this.size = size.copy();
    this.FBO = gl.createFramebuffer();

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.FBO);

    // create target textures
    let drawBuffers = [];
    for (let i = 0; i < this.attachments.length; i++) {
      this.attachments[i] = new Texture(gl, Texture.FLOAT, 4);
      drawBuffers.push(gl.COLOR_ATTACHMENT0 + i);
    }
    gl.drawBuffers(drawBuffers);

    for (let i = 0; i < this.attachments.length; i++) {
      gl.bindTexture(gl.TEXTURE_2D, this.attachments[i].id);
      this.attachments[i].resize(this.size);
  
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, this.attachments[i].id, 0);
    }
    this.depth = new Texture(gl, Texture.DEPTH);
    this.depth.resize(this.size);
    gl.bindTexture(gl.TEXTURE_2D, this.depth.id);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, this.depth.id, 0);

    console.log(`Framebuffer status: ${decodeFramebufferStatus(gl.checkFramebufferStatus(gl.FRAMEBUFFER))}`);
  } /* resize */

  bind() {
    let gl = this.gl;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.FBO);

    for (let i = 0; i < this.attachments.length; i++)
    gl.clearBufferfv(gl.COLOR, i, [0.00, 0.00, 0.00, 0.00]);
    gl.clearBufferfv(gl.DEPTH, 0, [1]);
    gl.viewport(0, 0, this.size.w, this.size.h);
  } /* bind */

  static defaultFramebuffer = {
    size: new Size(800, 600),
    gl: null,

    resize(size) {
      Target.defaultFramebuffer.size = size.copy();
    }, /* resize */

    bind() {
      let gl = Target.defaultFramebuffer.gl;

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, Target.defaultFramebuffer.size.w, Target.defaultFramebuffer.size.h);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.clearColor(0.30, 0.47, 0.80, 1.00);
    }
  }; /* defaultFramebuffer */

  static default(gl) {
    Target.defaultFramebuffer.gl = gl;
    return Target.defaultFramebuffer;
  } /* default */
} /* Target */

/* target.js */

class Render {
  renderQueue;
  gl;
  camera;
  cameraUBO;

  target;
  fsPrimitive = null;

  constructor() {
    // WebGL initialization
    let canvas = document.getElementById("glCanvas");

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    let gl = canvas.getContext("webgl2");
    if (gl == null) {
      throw Error("Can't initialize WebGL2");
    }

    let extensions = ["EXT_color_buffer_float", "OES_texture_float_linear"];
    for (let i = 0; i < extensions.length; i++)
      if (gl.getExtension(extensions[i]) == null)
        throw Error(`"${extensions[i]}" extension required`);

    this.gl = gl;

    gl.enable(WebGL2RenderingContext.DEPTH_TEST);

    this.renderQueue = [];
    this.camera = new Camera();

    this.cameraUBO = new UBO(this.gl);

    this.camera.resize(new Size(canvas.width, canvas.height));

    // targets setup
    let size = new Size(canvas.width, canvas.height);
    this.target = new Target(gl, 3);

    this.target.resize(size);
    Target.default(gl).resize(size);

    // resize handling
    window.onresize = () => {
      let resolution = new Size(window.innerWidth, window.innerHeight);

      canvas.width = resolution.w;
      canvas.height = resolution.h;

      this.camera.resize(resolution);
      this.target.resize(resolution);
      Target.default(gl).resize(resolution);
    };
  } /* constructor */

  drawPrimitive(primitive, transform = Mat4.identity()) {
    this.renderQueue.push({primitive, transform});
  } /* drawPrimitive */

  createTexture() {
    return new Texture(this.gl, Texture.UNSIGNED_BYTE, 4);
  } /* createTexture */

  createCubemap() {
    return new Cubemap(this.gl);
  } /* createCubemap */

  createUniformBuffer() {
    return new UBO(this.gl);
  } /* createUniformBuffer */

  async createShader(path) {
    return loadShader(this.gl, path);
  } /* createShader */

  async createMaterial(shader) {
    if (typeof(shader) == "string") {
      return new Material(this.gl, await loadShader(this.gl, shader));
    } else {
      return new Material(this.gl, shader);
    }
  } /* createMaterial */

  createPrimitive(topology, material) {
    return Primitive.fromTopology(this.gl, topology, material);
  } /* createPrimitive */

  async start() {
    if (this.fsPrimitive == null) {
      this.fsPrimitive = await this.createPrimitive(Topology.square(), await this.createMaterial("./bin/shaders/target"));
      this.fsPrimitive.material.textures = this.target.attachments;
    }
  } /* start */
  
  end() {
    // rendering in target
    let gl = this.gl;

    this.target.bind();

    let cameraInfo = new Float32Array(36);

    for (let i = 0; i < 16; i++) {
      cameraInfo[i + 16] = this.camera.viewProj.m[i];
    }
    cameraInfo[32] = this.camera.loc.x;
    cameraInfo[33] = this.camera.loc.y;
    cameraInfo[34] = this.camera.loc.z;

    for (let i = 0, count = this.renderQueue.length; i < count; i++) {
      let prim = this.renderQueue[i].primitive;
      let trans = this.renderQueue[i].transform;

      for (let i = 0; i < 16; i++) {
        cameraInfo[i] = trans.m[i];
      }

      this.cameraUBO.writeData(cameraInfo);

      prim.draw(this.cameraUBO);
    }

    // flush render queue
    this.renderQueue = [];

    // rendering to screen framebuffer
    Target.default(gl).bind();
    this.fsPrimitive.draw(this.cameraUBO);
  } /* end */
} /* Render */

/* render.js */

function getTime() {
  return Date.now() / 1000.0;
}

class Timer {
  fpsPrevUpdateTime = 0.00;
  startTime;
  fpsCounter = 0.00;
  pauseCollector = 0.00;
  isPaused = false;

  fpsDeltaTime = 3.00;
  fps = undefined;

  time = 0.00;
  globalTime;
  
  deltaTime = 0.00;
  globalDeltaTime = 0.00;

  constructor() {
    this.startTime = getTime();

    this.globalTime = this.startTime;
  } /* constructor */

  response() {
    let newGlobalTime = getTime();

    this.globalDeltaTime = newGlobalTime - this.globalTime;
    this.globalTime = newGlobalTime;

    if (this.isPaused) {
      this.deltaTime = 0.00;
      this.pauseCollector += this.globalDeltaTime;
    } else {
      this.time = this.globalTime - this.startTime - this.pauseCollector;
      this.deltaTime = this.globalDeltaTime;
    }

    this.fpsCounter++;
    if (this.globalTime - this.fpsPrevUpdateTime >= this.fpsDeltaTime) {
      this.fps = this.fpsCounter / (this.globalTime - this.fpsPrevUpdateTime);

      this.fpsPrevUpdateTime = this.globalTime;
      this.fpsCounter = 0;
    }
  } /* response */
} /* Timer */

/* timer.js */

class System {
  render; // Render
  units;  // unit list
  timer;  // timer

  constructor() {
    this.render = new Render();
    this.timer = new Timer();
    this.units = [];
  } /* constructor */

  static isPromise = function(v) {
    return v => typeof(v) === "object" && typeof v.then === "function";
  } /* isPromise */

  async addUnit(createFunction) {
    let val = createFunction(this);

    if (System.isPromise(val)) {
      val = await val;
    }

    this.units.push(val);

    return val;
  } /* addUnit */

  run() {
    let system = this;

    const run = async function() {
      system.timer.response();

      await system.render.start();

      for (let i = 0, count = system.units.length; i < count; i++) {
        system.units[i].response(system);
      }

      system.render.end();

      window.requestAnimationFrame(run);
    };
    window.requestAnimationFrame(run);
  } /* run */
} /* System */

// elevation of icosahedron basic sphere
const icosahedronElevation = 1.1071487177940904;

// icosahedron evaluation
let ico = new Topology();
ico.type = Topology.LINES;

ico.vtx = [];
ico.vtx.push(Vertex.fromCoord(0, 1, 0));
for (let a = 0; a < 5; a++) {
  ico.vtx.push(Vertex.fromVectors(Vec3.fromSpherical(a * Math.PI * 0.4, icosahedronElevation, 1)));
}
for (let a = 0; a < 5; a++) {
  ico.vtx.push(Vertex.fromVectors(Vec3.fromSpherical(a * Math.PI * 0.4 + Math.PI / 5.0, Math.PI - icosahedronElevation, 1)));
}
ico.vtx.push(Vertex.fromCoord(0, -1, 0));

// write wire indices
ico.idx = [];
for (let a = 0; a < 5; a++) {
  ico.idx.push(0);
  ico.idx.push(a + 1);

  ico.idx.push(a + 1);
  ico.idx.push((a + 1) % 5 + 1);

  ico.idx.push((a + 1) % 5 + 1);
  ico.idx.push(a + 6);

  ico.idx.push(a + 1);
  ico.idx.push(a + 6);

  ico.idx.push(a + 6);
  ico.idx.push((a + 1) % 5 + 6);

  ico.idx.push(11);
  ico.idx.push(a + 6);
}

// write polygonal indices
ico.polygon_idx = [];
for (let y = 0; y < 5; y++) {
  ico.polygon_idx.push(0);
  ico.polygon_idx.push(y + 1);
  ico.polygon_idx.push((y + 1) % 5 + 1);

  ico.polygon_idx.push(y + 1);
  ico.polygon_idx.push((y + 1) % 5 + 1);
  ico.polygon_idx.push(y + 6);

  ico.polygon_idx.push(y + 6);
  ico.polygon_idx.push((y + 1) % 5 + 1);
  ico.polygon_idx.push((y + 1) % 5 + 6);

  ico.polygon_idx.push((y + 1) % 5 + 6);
  ico.polygon_idx.push(y + 6);
  ico.polygon_idx.push(11);
}


// dodecahedron evaluation
let dod = new Topology([], []);
dod.type = Topology.LINES;

for (let i = 0; i < ico.polygon_idx.length / 3; i++) {
  let p1 = ico.vtx[ico.polygon_idx[i * 3 + 0]].p;
  let p2 = ico.vtx[ico.polygon_idx[i * 3 + 1]].p;
  let p3 = ico.vtx[ico.polygon_idx[i * 3 + 2]].p;

  dod.vtx.push(Vertex.fromVectors(p1.add(p2).add(p3).mul(1.0 / 3)));
}

for (let a = 0; a < 5; a++) {
  dod.idx.push(a * 4 + 0);
  dod.idx.push(a * 4 + 1);

  dod.idx.push(a * 4 + 1);
  dod.idx.push(a * 4 + 2);

  dod.idx.push(a * 4 + 2);
  dod.idx.push(a * 4 + 3);

  dod.idx.push(a * 4 + 0);
  dod.idx.push((a + 1) % 5 * 4 + 0);

  dod.idx.push(a * 4 + 2);
  dod.idx.push((a + 1) % 5 * 4 + 1);

  dod.idx.push(a * 4 + 3);
  dod.idx.push((a + 1) % 5 * 4 + 3);
}

// cube evaluation
let cub = new Topology();
cub.type = Topology.LINES;
cub.vtx = [
  Vertex.fromCoord(-1, -1, -1),
  Vertex.fromCoord(-1,  1, -1),
  Vertex.fromCoord( 1,  1, -1),
  Vertex.fromCoord( 1, -1, -1),

  Vertex.fromCoord(-1, -1,  1),
  Vertex.fromCoord(-1,  1,  1),
  Vertex.fromCoord( 1,  1,  1),
  Vertex.fromCoord( 1, -1,  1),
];

cub.idx = [
  0, 1, 1, 2, 2, 3, 3, 0,
  4, 5, 5, 6, 6, 7, 7, 4,
  0, 4, 1, 5, 2, 6, 3, 7
];

// octahedron evaluation
let oct = new Topology();
oct.type = Topology.LINES;

oct.vtx = [
  Vertex.fromCoord( 0,  1,  0),

  Vertex.fromCoord( 1,  0,  0),
  Vertex.fromCoord( 0,  0,  1),
  Vertex.fromCoord(-1,  0,  0),
  Vertex.fromCoord( 0,  0, -1),

  Vertex.fromCoord( 0, -1,  0),
];
oct.idx = [
  0, 1, 0, 2, 0, 3, 0, 4,
  1, 2, 2, 3, 3, 4, 4, 1,
  5, 1, 5, 2, 5, 3, 5, 4
];

// tetrahedron evalutation
const tetHeight = Math.sqrt(2 / 3);
const tetInnerLen = 1 / Math.sqrt(3);

let tet = new Topology();
tet.type = Topology.LINES;
tet.vtx = [Vertex.fromVectors(new Vec3(0, tetHeight, 0))];
for (let a = 0; a < 3; a++) {
  tet.vtx.push(Vertex.fromVectors(new Vec3(
    Math.cos(2 * Math.PI / 3 * a) * tetInnerLen * 2,
    -tetHeight,
    Math.sin(2 * Math.PI / 3 * a) * tetInnerLen * 2
  )));
}
tet.idx = [0, 1, 0, 2, 0, 3, 1, 2, 2, 3, 3, 1];

// add platonic bodies units
async function create(system) {
  let platonicMtl = await system.render.createMaterial("./bin/shaders/platonic");

  let prims = [
    await system.render.createPrimitive(ico, platonicMtl),
    await system.render.createPrimitive(dod, platonicMtl),
    await system.render.createPrimitive(cub, platonicMtl),
    await system.render.createPrimitive(oct, platonicMtl),
    await system.render.createPrimitive(tet, platonicMtl)
  ];

  return {
    response(system) {
      for (let i = 0; i < prims.length; i++) {
        system.render.drawPrimitive(prims[i],
              Mat4.rotateY(system.timer.time).
          mul(Mat4.translate(new Vec3(i * 2.40, 0, 0)))
        );
      }
    }
  };
} /* create */

/* platonic_bodies.js */

let system = new System();

document.onkeydown = (event) => {
  if (event.key != 'P')
    return;

  system.addUnit(async function() {
    let tpl = Topology.square();
    let mtl = await system.render.createMaterial("./bin/shaders/skysphere");

    let tex = system.render.createTexture();
    tex.load("./bin/skyspheres/sky12.png");
    let gl = system.render.gl;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTUER_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR_MIPMAP_LINEAR);

    mtl.textures.push(tex);
    mtl.ubo = system.render.createUniformBuffer();
    mtl.uboNameOnShader = "projectionInfo";

    let prim = await system.render.createPrimitive(tpl, mtl);

    return {
      response(system) {
        let dir = system.render.camera.dir.mul(system.render.camera.near);
        let rgh = system.render.camera.right.mul(system.render.camera.projSize.w);
        let tup = system.render.camera.up.mul(system.render.camera.projSize.h);
        
        mtl.ubo.writeData(new Float32Array([
          dir.x, dir.y, dir.z, 0,
          rgh.x, rgh.y, rgh.z, 0,
          tup.x, tup.y, tup.z, 0
        ]));

        system.render.drawPrimitive(prim);
      }
    };
  });
};

system.addUnit(create);

system.addUnit(async function() {
    let tpl = await Topology.model_obj("./bin/models/headcrab/headcrab.obj");
    let mtl = await system.render.createMaterial("./bin/shaders/default");
    let tex = system.render.createTexture();
  tex.load("./bin/models/headcrab/diffuse.png");
  mtl.textures.push(tex);

  let prim = await system.render.createPrimitive(tpl, mtl);
  let transform = Mat4.translate(new Vec3(5, 0, -12));

  return {
    response(system) {
      system.render.drawPrimitive(prim, transform);
    }
  };
});

// camera unit
system.addUnit(function() {
  const up = new Vec3(0, 1, 0);
  let loc = new Vec3(19.88, 11.67, 9.20), at = new Vec3(8.90, 5.32, -4.65);
  let radius = at.sub(loc).length();

  let camera = {
    response(system) {
      system.render.camera.set(loc, at, up);
    } /* response */
  };

  const onMouseMove = function(event) {
    if ((event.buttons & 1) == 1) { // rotate
      let direction = loc.sub(at);

      // turn direction to polar coordinate system
      radius = direction.length();
      let
        azimuth  = Math.sign(direction.z) * Math.acos(direction.x / Math.sqrt(direction.x * direction.x + direction.z * direction.z)),
        elevator = Math.acos(direction.y / direction.length());

      // rotate direction
      azimuth  += event.movementX / 200.0;
      elevator -= event.movementY / 200.0;

      elevator = Math.min(Math.max(elevator, 0.01), Math.PI);

       // restore direction
      direction.x = radius * Math.sin(elevator) * Math.cos(azimuth);
      direction.y = radius * Math.cos(elevator);
      direction.z = radius * Math.sin(elevator) * Math.sin(azimuth);

      loc = at.add(direction);
    }

    if ((event.buttons & 2) == 2) { // move
      let dir = at.sub(loc).normalize();
      let rgh = dir.cross(up).normalize();
      let tup = rgh.cross(dir);

      let delta = rgh.mul(-event.movementX * radius / 300.0).add(tup.mul(event.movementY * radius / 300.0));
      loc = loc.add(delta);
      at = at.add(delta);
    }
  };

  const onWheel = function(event) {
    let delta = event.deltaY / 100.0;

    loc = loc.add(at.sub(loc).mul(delta * system.timer.deltaTime));
  };

  let canvas = document.getElementById("glCanvas");

  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("wheel", onWheel);

  return camera;
});

// // FPS controller unit
// system.addUnit(function() {
//   let u = {
//     text: document.getElementById("FPS"),
//     response: function() {
//       u.text.innerText = `FPS: ${system.timer.fps}`;
//     }
//   };

//   return u;
// });

system.run();

/* main.js */
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYnVuZGxlLmpzIiwic291cmNlcyI6WyJzcmMvc2hhZGVyLmpzIiwic3JjL210aC5qcyIsInNyYy90ZXh0dXJlLmpzIiwic3JjL3Viby5qcyIsInNyYy9tYXRlcmlhbC5qcyIsInNyYy9wcmltaXRpdmUuanMiLCJzcmMvdGFyZ2V0LmpzIiwic3JjL3JlbmRlci5qcyIsInNyYy90aW1lci5qcyIsInNyYy9zeXN0ZW0uanMiLCJzcmMvdW5pdHMvcGxhdG9uaWNfYm9kaWVzLmpzIiwic3JjL21haW4uanMiXSwic291cmNlc0NvbnRlbnQiOlsiYXN5bmMgZnVuY3Rpb24gbG9hZFNoYWRlck1vZHVsZShnbCwgdHlwZSwgcGF0aCkge1xyXG4gIGxldCBzcmMgPSBhd2FpdCBmZXRjaChwYXRoKS50aGVuKHJlc3BvbnNlID0+IHJlc3BvbnNlLnRleHQoKSk7XHJcbiAgbGV0IHNoYWRlciA9IGdsLmNyZWF0ZVNoYWRlcih0eXBlKTtcclxuXHJcbiAgZ2wuc2hhZGVyU291cmNlKHNoYWRlciwgc3JjKTtcclxuICBnbC5jb21waWxlU2hhZGVyKHNoYWRlcik7XHJcblxyXG4gIGxldCByZXMgPSBnbC5nZXRTaGFkZXJJbmZvTG9nKHNoYWRlcik7XHJcbiAgaWYgKHJlcyAhPSBudWxsICYmIHJlcy5sZW5ndGggPiAwKVxyXG4gICAgY29uc29sZS5lcnJvcihgU2hhZGVyIG1vZHVsZSBjb21waWxhdGlvbiBlcnJvcjogJHtyZXN9YCk7XHJcblxyXG4gIHJldHVybiBhd2FpdCBzaGFkZXJNb2R1bGVGcm9tU291cmNlKGdsLCB0eXBlLCBhd2FpdCBmZXRjaChwYXRoKS50aGVuKChyZXNwb25zZSwgb2spID0+IG9rID8gcmVzcG9uc2UgOiBudWxsKSk7XHJcbn0gLyogbG9hZFNoYWRlck1vZHVsZSAqL1xyXG5cclxuZnVuY3Rpb24gc2hhZGVyTW9kdWxlRnJvbVNvdXJjZShnbCwgdHlwZSwgc291cmNlKSB7XHJcbiAgaWYgKHNvdXJjZSA9PSBudWxsKSB7XHJcbiAgICByZXR1cm4gbnVsbDtcclxuICB9XHJcblxyXG4gIGxldCBzaGFkZXIgPSBnbC5jcmVhdGVTaGFkZXIodHlwZSk7XHJcblxyXG4gIGdsLnNoYWRlclNvdXJjZShzaGFkZXIsIHNvdXJjZSk7XHJcbiAgZ2wuY29tcGlsZVNoYWRlcihzaGFkZXIpO1xyXG5cclxuICBsZXQgcmVzID0gZ2wuZ2V0U2hhZGVySW5mb0xvZyhzaGFkZXIpO1xyXG4gIGlmIChyZXMgIT0gbnVsbCAmJiByZXMubGVuZ3RoID4gMClcclxuICAgIGNvbnNvbGUuZXJyb3IoYFNoYWRlciBtb2R1bGUgY29tcGlsYXRpb24gZXJyb3I6ICR7cmVzfWApO1xyXG5cclxuICByZXR1cm4gc2hhZGVyO1xyXG59IC8qIHNoYWRlck1vZHVsZUZyb21Tb3VyY2UgKi9cclxuXHJcbmV4cG9ydCBmdW5jdGlvbiBzaGFkZXJGcm9tU291cmNlKGdsLCB2ZXJ0U291cmNlLCBnZW9tU291cmNlLCBmcmFnU291cmNlKSB7XHJcbiAgbGV0IG1vZHVsZXMgPSBbXHJcbiAgICBzaGFkZXJNb2R1bGVGcm9tU291cmNlKGdsLCBnbC5WRVJURVhfU0hBREVSLCB2ZXJ0U291cmNlKSxcclxuICAgIHNoYWRlck1vZHVsZUZyb21Tb3VyY2UoZ2wsIGdsLkdFT01FVFJZX1NIQURFUiwgZ2VvbVNvdXJjZSksXHJcbiAgICBzaGFkZXJNb2R1bGVGcm9tU291cmNlKGdsLCBnbC5GUkFHTUVOVF9TSEFERVIsIGZyYWdTb3VyY2UpLFxyXG4gIF07XHJcbiAgbGV0IHByb2dyYW0gPSBnbC5jcmVhdGVQcm9ncmFtKCk7XHJcblxyXG4gIGZvciAobGV0IGkgPSAwOyBpIDwgMzsgaSsrKSB7XHJcbiAgICBpZiAobW9kdWxlc1tpXSAhPSBudWxsKSB7XHJcbiAgICAgIGdsLmF0dGFjaFNoYWRlcihwcm9ncmFtLCBtb2R1bGVzW2ldKTtcclxuICAgIH1cclxuICB9XHJcblxyXG4gIGdsLmxpbmtQcm9ncmFtKHByb2dyYW0pO1xyXG5cclxuICBpZiAoIWdsLmdldFByb2dyYW1QYXJhbWV0ZXIocHJvZ3JhbSwgZ2wuTElOS19TVEFUVVMpKVxyXG4gICAgY29uc29sZS5lcnJvcihgU2hhZGVyIGxpbmtpbmcgZXJyb3I6ICR7Z2wuZ2V0UHJvZ3JhbUluZm9Mb2cocHJvZ3JhbSl9YCk7XHJcblxyXG4gIHJldHVybiBwcm9ncmFtO1xyXG59IC8qIHNoYWRlckZyb21Tb3VyY2UgKi9cclxuXHJcbmV4cG9ydCBhc3luYyBmdW5jdGlvbiBsb2FkU2hhZGVyKGdsLCBwYXRoKSB7XHJcbiAgcmV0dXJuIHNoYWRlckZyb21Tb3VyY2UoZ2wsXHJcbiAgICBhd2FpdCBmZXRjaChwYXRoICsgXCIudmVydD9cIiArIE1hdGgucmFuZG9tKCkudG9TdHJpbmcoKSkudGhlbihyZXNwb25zZSA9PiByZXNwb25zZS5vayA/IHJlc3BvbnNlLnRleHQoKSA6IG51bGwpLFxyXG4gICAgYXdhaXQgZmV0Y2gocGF0aCArIFwiLmdlb20/XCIgKyBNYXRoLnJhbmRvbSgpLnRvU3RyaW5nKCkpLnRoZW4ocmVzcG9uc2UgPT4gcmVzcG9uc2Uub2sgPyByZXNwb25zZS50ZXh0KCkgOiBudWxsKSxcclxuICAgIGF3YWl0IGZldGNoKHBhdGggKyBcIi5mcmFnP1wiICsgTWF0aC5yYW5kb20oKS50b1N0cmluZygpKS50aGVuKHJlc3BvbnNlID0+IHJlc3BvbnNlLm9rID8gcmVzcG9uc2UudGV4dCgpIDogbnVsbCksXHJcbiAgKTtcclxufSAvKiBsb2FkU2hhZGVyICovIiwiZXhwb3J0IGNsYXNzIFZlYzMge1xyXG4gIHg7XHJcbiAgeTtcclxuICB6O1xyXG5cclxuICBjb25zdHJ1Y3RvcihueCwgbnksIG56KSB7XHJcbiAgICB0aGlzLnggPSBueDtcclxuICAgIHRoaXMueSA9IG55O1xyXG4gICAgdGhpcy56ID0gbno7XHJcbiAgfSAvKiBjb25zdHJ1Y3RvciAqL1xyXG5cclxuICBjb3B5KCkge1xyXG4gICAgcmV0dXJuIG5ldyBWZWMzKHRoaXMueCwgdGhpcy55LCB0aGlzLnopO1xyXG4gIH0gLyogY29weSAqL1xyXG5cclxuICBhZGQobTIpIHtcclxuICAgIHJldHVybiBuZXcgVmVjMyh0aGlzLnggKyBtMi54LCB0aGlzLnkgKyBtMi55LCB0aGlzLnogKyBtMi56KTtcclxuICB9IC8qIGFkZCAqL1xyXG5cclxuICBzdWIobTIpIHtcclxuICAgIHJldHVybiBuZXcgVmVjMyh0aGlzLnggLSBtMi54LCB0aGlzLnkgLSBtMi55LCB0aGlzLnogLSBtMi56KTtcclxuICB9IC8qIHN1YiAqL1xyXG5cclxuICBtdWwobTIpIHtcclxuICAgIGlmICh0eXBlb2YobTIpID09IFwiVmVjM1wiKVxyXG4gICAgICByZXR1cm4gbmV3IFZlYzModGhpcy54ICogbTIueCwgdGhpcy55ICogbTIueSwgdGhpcy56ICogbTIueik7XHJcbiAgICBlbHNlXHJcbiAgICAgIHJldHVybiBuZXcgVmVjMyh0aGlzLnggKiBtMiwgICB0aGlzLnkgKiBtMiwgICB0aGlzLnogKiBtMiAgKTtcclxuICB9IC8qIG11bCAqL1xyXG5cclxuICBsZW5ndGgoKSB7XHJcbiAgICByZXR1cm4gTWF0aC5zcXJ0KHRoaXMueCAqIHRoaXMueCArIHRoaXMueSAqIHRoaXMueSArIHRoaXMueiAqIHRoaXMueik7XHJcbiAgfSAvKiBsZW5ndGggKi9cclxuICBcclxuICBkb3QobTIpIHtcclxuICAgIHJldHVybiB0aGlzLnggKiBtMi54ICsgdGhpcy55ICogbTIueSArIHRoaXMueiAqIG0yLno7XHJcbiAgfSAvKiBkb3QgKi9cclxuXHJcbiAgY3Jvc3Mob3Rodikge1xyXG4gICAgcmV0dXJuIG5ldyBWZWMzKFxyXG4gICAgICB0aGlzLnkgKiBvdGh2LnogLSBvdGh2LnkgKiB0aGlzLnosXHJcbiAgICAgIHRoaXMueiAqIG90aHYueCAtIG90aHYueiAqIHRoaXMueCxcclxuICAgICAgdGhpcy54ICogb3Rodi55IC0gb3Rodi54ICogdGhpcy55XHJcbiAgICApO1xyXG4gIH0gLyogY3Jvc3MgKi9cclxuXHJcbiAgbm9ybWFsaXplKCkge1xyXG4gICAgbGV0IGxlbiA9IHRoaXMubGVuZ3RoKCk7XHJcblxyXG4gICAgcmV0dXJuIG5ldyBWZWMzKHRoaXMueCAvIGxlbiwgdGhpcy55IC8gbGVuLCB0aGlzLnogLyBsZW4pO1xyXG4gIH0gLyogbm9ybWFsaXplICovXHJcblxyXG4gIHN0YXRpYyBmcm9tU3BoZXJpY2FsKGF6aW11dGgsIGVsZXZhdGlvbiwgcmFkaXVzID0gMSkge1xyXG4gICAgcmV0dXJuIG5ldyBWZWMzKFxyXG4gICAgICByYWRpdXMgKiBNYXRoLnNpbihlbGV2YXRpb24pICogTWF0aC5jb3MoYXppbXV0aCksXHJcbiAgICAgIHJhZGl1cyAqIE1hdGguY29zKGVsZXZhdGlvbiksXHJcbiAgICAgIHJhZGl1cyAqIE1hdGguc2luKGVsZXZhdGlvbikgKiBNYXRoLnNpbihhemltdXRoKVxyXG4gICAgKTtcclxuICB9IC8qIHNwaGVyaWNhbFRvQ2FydGVzaWFuICovXHJcbn0gLyogVmVjMyAqL1xyXG5cclxuZXhwb3J0IGNsYXNzIFZlYzIge1xyXG4gIHg7XHJcbiAgeTtcclxuXHJcbiAgY29uc3RydWN0b3IobngsIG55KSB7XHJcbiAgICB0aGlzLnggPSBueDtcclxuICAgIHRoaXMueSA9IG55O1xyXG4gIH0gLyogY29uc3RydWN0b3IgKi9cclxuXHJcbiAgY29weSgpIHtcclxuICAgIHJldHVybiBuZXcgVmVjMigpO1xyXG4gIH1cclxuXHJcbiAgYWRkKG0yKSB7XHJcbiAgICByZXR1cm4gbmV3IFZlYzIodGhpcy54ICsgbTIueCwgdGhpcy55ICsgbTIueSk7XHJcbiAgfSAvKiBhZGQgKi9cclxuXHJcbiAgc3ViKG0yKSB7XHJcbiAgICByZXR1cm4gbmV3IFZlYzIodGhpcy54IC0gbTIueCwgdGhpcy55IC0gbTIueSk7XHJcbiAgfSAvKiBzdWIgKi9cclxuXHJcbiAgbXVsKG0yKSB7XHJcbiAgICBpZiAodHlwZW9mKG0yKSA9PSBcIlZlYzJcIilcclxuICAgICAgcmV0dXJuIG5ldyBWZWMyKHRoaXMueCAqIG0yLngsIHRoaXMueSAqIG0yLnkpO1xyXG4gICAgZWxzZVxyXG4gICAgICByZXR1cm4gbmV3IFZlYzIodGhpcy54ICogbTIsICAgdGhpcy55ICogbTIpO1xyXG4gIH0gLyogbXVsICovXHJcblxyXG4gIGxlbmd0aDIoKSB7XHJcbiAgICByZXR1cm4gdGhpcy54ICogdGhpcy54ICsgdGhpcy55ICogdGhpczt5XHJcbiAgfSAvKiBsZW5ndGgyICovXHJcblxyXG4gIGxlbmd0aCgpIHtcclxuICAgIHJldHVybiBNYXRoLnNxcnQodGhpcy54ICogdGhpcy54ICsgdGhpcy55ICogdGhpcy55KTtcclxuICB9IC8qIGxlbmd0aCAqL1xyXG5cclxuICBkb3QobTIpIHtcclxuICAgIHJldHVybiB0aGlzLnggKiBtMi54ICsgdGhpcy55ICogbTIueTtcclxuICB9IC8qIGRvdCAqL1xyXG5cclxuICBjcm9zcyhvdGh2KSB7XHJcbiAgICByZXR1cm4gdGhpcy54ICogb3Rodi55IC0gb3Rodi54ICogdGhpcy55O1xyXG4gIH0gLyogY3Jvc3MgKi9cclxuXHJcbiAgbm9ybWFsaXplKCkge1xyXG4gICAgbGV0IGxlbiA9IHRoaXMubGVuZ3RoKCk7XHJcblxyXG4gICAgcmV0dXJuIG5ldyBWZWMyKHRoaXMueCAvIGxlbiwgdGhpcy55IC8gbGVuKTtcclxuICB9IC8qIG5vcm1hbGl6ZSAqL1xyXG59IC8qIFZlYzIgKi9cclxuXHJcbmV4cG9ydCBjbGFzcyBTaXplIHtcclxuICB3O1xyXG4gIGg7XHJcblxyXG4gIGNvbnN0cnVjdG9yKHcsIGgpIHtcclxuICAgIHRoaXMudyA9IHc7XHJcbiAgICB0aGlzLmggPSBoO1xyXG4gIH0gLyogY29uc3RydWN0b3IgKi9cclxuXHJcbiAgY29weSgpIHtcclxuICAgIHJldHVybiBuZXcgU2l6ZSh0aGlzLncsIHRoaXMuaCk7XHJcbiAgfSAvKiBjb3B5ICovXHJcbn0gLyogU2l6ZSAqL1xyXG5cclxuZXhwb3J0IGNsYXNzIE1hdDQge1xyXG4gIG07XHJcblxyXG4gIGNvbnN0cnVjdG9yKHYwMCwgdjAxLCB2MDIsIHYwMyxcclxuICAgICAgICAgICAgICB2MTAsIHYxMSwgdjEyLCB2MTMsXHJcbiAgICAgICAgICAgICAgdjIwLCB2MjEsIHYyMiwgdjIzLFxyXG4gICAgICAgICAgICAgIHYzMCwgdjMxLCB2MzIsIHYzMykge1xyXG4gICAgdGhpcy5tID0gW1xyXG4gICAgICB2MDAsIHYwMSwgdjAyLCB2MDMsXHJcbiAgICAgIHYxMCwgdjExLCB2MTIsIHYxMyxcclxuICAgICAgdjIwLCB2MjEsIHYyMiwgdjIzLFxyXG4gICAgICB2MzAsIHYzMSwgdjMyLCB2MzNcclxuICAgIF07XHJcbiAgfSAvKiBjb25zdHJ1Y3RvciAqL1xyXG5cclxuICBjb3B5KCkge1xyXG4gICAgcmV0dXJuIG5ldyBNYXQ0KFxyXG4gICAgICB0aGlzLm1bIDBdLCB0aGlzLm1bIDFdLCB0aGlzLm1bIDJdLCB0aGlzLm1bIDNdLFxyXG4gICAgICB0aGlzLm1bIDRdLCB0aGlzLm1bIDVdLCB0aGlzLm1bIDZdLCB0aGlzLm1bIDddLFxyXG4gICAgICB0aGlzLm1bIDhdLCB0aGlzLm1bIDldLCB0aGlzLm1bMTBdLCB0aGlzLm1bMTFdLFxyXG4gICAgICB0aGlzLm1bMTJdLCB0aGlzLm1bMTNdLCB0aGlzLm1bMTRdLCB0aGlzLm1bMTVdLFxyXG4gICAgKTtcclxuICB9IC8qIGNvcHkgKi9cclxuXHJcbiAgdHJhbnNmb3JtUG9pbnQodilcclxuICB7XHJcbiAgICByZXR1cm4gbmV3IFZlYzMoXHJcbiAgICAgIHYueCAqIHRoaXMubVsgMF0gKyB2LnkgKiB0aGlzLm1bIDRdICsgdi56ICogdGhpcy5tWyA4XSArIHRoaXMubVsxMl0sXHJcbiAgICAgIHYueCAqIHRoaXMubVsgMV0gKyB2LnkgKiB0aGlzLm1bIDVdICsgdi56ICogdGhpcy5tWyA5XSArIHRoaXMubVsxM10sXHJcbiAgICAgIHYueCAqIHRoaXMubVsgMl0gKyB2LnkgKiB0aGlzLm1bIDZdICsgdi56ICogdGhpcy5tWzEwXSArIHRoaXMubVsxNF1cclxuICAgICk7XHJcbiAgfSAvKiB0cmFuc2Zvcm1Qb2ludCAqL1xyXG5cclxuICB0cmFuc2Zvcm00eDQodilcclxuICB7XHJcbiAgICBsZXQgdyA9IFYueCAqIHRoaXMubVszXSArIFYueSAqIHRoaXMubVs3XSArIFYueiAqIHRoaXMubVsxMV0gKyB0aGlzLm1bMTVdO1xyXG4gIFxyXG4gICAgcmV0dXJuIG5ldyBWZWMzKFxyXG4gICAgICAodi54ICogdGhpcy5tWyAwXSArIHYueSAqIHRoaXMubVsgNF0gKyB2LnogKiB0aGlzLm1bIDhdICsgdGhpcy5tWzEyXSkgLyB3LFxyXG4gICAgICAodi54ICogdGhpcy5tWyAxXSArIHYueSAqIHRoaXMubVsgNV0gKyB2LnogKiB0aGlzLm1bIDldICsgdGhpcy5tWzEzXSkgLyB3LFxyXG4gICAgICAodi54ICogdGhpcy5tWyAyXSArIHYueSAqIHRoaXMubVsgNl0gKyB2LnogKiB0aGlzLm1bMTBdICsgdGhpcy5tWzE0XSkgLyB3XHJcbiAgICApO1xyXG4gIH0gLyogdHJhbnNmb3JtNHg0ICovXHJcblxyXG4gIHRyYW5zcG9zZSgpIHtcclxuICAgIHJldHVybiBuZXcgTWF0NChcclxuICAgICAgdGhpcy5tWyAwXSwgdGhpcy5tWyA0XSwgdGhpcy5tWyA4XSwgdGhpcy5tWzEyXSxcclxuICAgICAgdGhpcy5tWyAxXSwgdGhpcy5tWyA1XSwgdGhpcy5tWyA5XSwgdGhpcy5tWzEzXSxcclxuICAgICAgdGhpcy5tWyAyXSwgdGhpcy5tWyA2XSwgdGhpcy5tWzEwXSwgdGhpcy5tWzE0XSxcclxuICAgICAgdGhpcy5tWyAzXSwgdGhpcy5tWyA3XSwgdGhpcy5tWzExXSwgdGhpcy5tWzE1XVxyXG4gICAgKTtcclxuICB9IC8qIHRyYW5zcG9zZSAqL1xyXG5cclxuICBtdWwobTIpIHtcclxuICAgIHJldHVybiBuZXcgTWF0NChcclxuICAgICAgdGhpcy5tWyAwXSAqIG0yLm1bIDBdICsgdGhpcy5tWyAxXSAqIG0yLm1bIDRdICsgdGhpcy5tWyAyXSAqIG0yLm1bIDhdICsgdGhpcy5tWyAzXSAqIG0yLm1bMTJdLFxyXG4gICAgICB0aGlzLm1bIDBdICogbTIubVsgMV0gKyB0aGlzLm1bIDFdICogbTIubVsgNV0gKyB0aGlzLm1bIDJdICogbTIubVsgOV0gKyB0aGlzLm1bIDNdICogbTIubVsxM10sXHJcbiAgICAgIHRoaXMubVsgMF0gKiBtMi5tWyAyXSArIHRoaXMubVsgMV0gKiBtMi5tWyA2XSArIHRoaXMubVsgMl0gKiBtMi5tWzEwXSArIHRoaXMubVsgM10gKiBtMi5tWzE0XSxcclxuICAgICAgdGhpcy5tWyAwXSAqIG0yLm1bIDNdICsgdGhpcy5tWyAxXSAqIG0yLm1bIDddICsgdGhpcy5tWyAyXSAqIG0yLm1bMTFdICsgdGhpcy5tWyAzXSAqIG0yLm1bMTVdLFxyXG5cclxuICAgICAgdGhpcy5tWyA0XSAqIG0yLm1bIDBdICsgdGhpcy5tWyA1XSAqIG0yLm1bIDRdICsgdGhpcy5tWyA2XSAqIG0yLm1bIDhdICsgdGhpcy5tWyA3XSAqIG0yLm1bMTJdLFxyXG4gICAgICB0aGlzLm1bIDRdICogbTIubVsgMV0gKyB0aGlzLm1bIDVdICogbTIubVsgNV0gKyB0aGlzLm1bIDZdICogbTIubVsgOV0gKyB0aGlzLm1bIDddICogbTIubVsxM10sXHJcbiAgICAgIHRoaXMubVsgNF0gKiBtMi5tWyAyXSArIHRoaXMubVsgNV0gKiBtMi5tWyA2XSArIHRoaXMubVsgNl0gKiBtMi5tWzEwXSArIHRoaXMubVsgN10gKiBtMi5tWzE0XSxcclxuICAgICAgdGhpcy5tWyA0XSAqIG0yLm1bIDNdICsgdGhpcy5tWyA1XSAqIG0yLm1bIDddICsgdGhpcy5tWyA2XSAqIG0yLm1bMTFdICsgdGhpcy5tWyA3XSAqIG0yLm1bMTVdLFxyXG5cclxuICAgICAgdGhpcy5tWyA4XSAqIG0yLm1bIDBdICsgdGhpcy5tWyA5XSAqIG0yLm1bIDRdICsgdGhpcy5tWzEwXSAqIG0yLm1bIDhdICsgdGhpcy5tWzExXSAqIG0yLm1bMTJdLFxyXG4gICAgICB0aGlzLm1bIDhdICogbTIubVsgMV0gKyB0aGlzLm1bIDldICogbTIubVsgNV0gKyB0aGlzLm1bMTBdICogbTIubVsgOV0gKyB0aGlzLm1bMTFdICogbTIubVsxM10sXHJcbiAgICAgIHRoaXMubVsgOF0gKiBtMi5tWyAyXSArIHRoaXMubVsgOV0gKiBtMi5tWyA2XSArIHRoaXMubVsxMF0gKiBtMi5tWzEwXSArIHRoaXMubVsxMV0gKiBtMi5tWzE0XSxcclxuICAgICAgdGhpcy5tWyA4XSAqIG0yLm1bIDNdICsgdGhpcy5tWyA5XSAqIG0yLm1bIDddICsgdGhpcy5tWzEwXSAqIG0yLm1bMTFdICsgdGhpcy5tWzExXSAqIG0yLm1bMTVdLFxyXG5cclxuICAgICAgdGhpcy5tWzEyXSAqIG0yLm1bIDBdICsgdGhpcy5tWzEzXSAqIG0yLm1bIDRdICsgdGhpcy5tWzE0XSAqIG0yLm1bIDhdICsgdGhpcy5tWzE1XSAqIG0yLm1bMTJdLFxyXG4gICAgICB0aGlzLm1bMTJdICogbTIubVsgMV0gKyB0aGlzLm1bMTNdICogbTIubVsgNV0gKyB0aGlzLm1bMTRdICogbTIubVsgOV0gKyB0aGlzLm1bMTVdICogbTIubVsxM10sXHJcbiAgICAgIHRoaXMubVsxMl0gKiBtMi5tWyAyXSArIHRoaXMubVsxM10gKiBtMi5tWyA2XSArIHRoaXMubVsxNF0gKiBtMi5tWzEwXSArIHRoaXMubVsxNV0gKiBtMi5tWzE0XSxcclxuICAgICAgdGhpcy5tWzEyXSAqIG0yLm1bIDNdICsgdGhpcy5tWzEzXSAqIG0yLm1bIDddICsgdGhpcy5tWzE0XSAqIG0yLm1bMTFdICsgdGhpcy5tWzE1XSAqIG0yLm1bMTVdLFxyXG4gICAgKTtcclxuICB9IC8qIG11bCAqL1xyXG5cclxuICBzdGF0aWMgaWRlbnRpdHkoKSB7XHJcbiAgICByZXR1cm4gbmV3IE1hdDQoXHJcbiAgICAgIDEsIDAsIDAsIDAsXHJcbiAgICAgIDAsIDEsIDAsIDAsXHJcbiAgICAgIDAsIDAsIDEsIDAsXHJcbiAgICAgIDAsIDAsIDAsIDFcclxuICAgICk7XHJcbiAgfSAvKiBpZGVudGl0eSAqL1xyXG5cclxuICBzdGF0aWMgc2NhbGUocykge1xyXG4gICAgcmV0dXJuIG5ldyBNYXQ0KFxyXG4gICAgICBzLngsIDAsICAgMCwgICAwLFxyXG4gICAgICAwLCAgIHMueSwgMCwgICAwLFxyXG4gICAgICAwLCAgIDAsICAgcy56LCAwLFxyXG4gICAgICAwLCAgIDAsICAgMCwgICAxXHJcbiAgICApO1xyXG4gIH0gLyogc2NhbGUgKi9cclxuXHJcbiAgc3RhdGljIHRyYW5zbGF0ZSh0KSB7XHJcbiAgICByZXR1cm4gbmV3IE1hdDQoXHJcbiAgICAgIDEsICAgMCwgICAwLCAgIDAsXHJcbiAgICAgIDAsICAgMSwgICAwLCAgIDAsXHJcbiAgICAgIDAsICAgMCwgICAxLCAgIDAsXHJcbiAgICAgIHQueCwgdC55LCB0LnosIDFcclxuICAgICk7XHJcbiAgfSAvKiB0cmFuc2xhdGUgKi9cclxuXHJcbiAgc3RhdGljIHJvdGF0ZVgoYW5nbGUpIHtcclxuICAgIGxldCBzID0gTWF0aC5zaW4oYW5nbGUpLCBjID0gTWF0aC5jb3MoYW5nbGUpO1xyXG5cclxuICAgIHJldHVybiBuZXcgTWF0NChcclxuICAgICAgMSwgMCwgMCwgMCxcclxuICAgICAgMCwgYywgcywgMCxcclxuICAgICAgMCwtcywgYywgMCxcclxuICAgICAgMCwgMCwgMCwgMVxyXG4gICAgKTtcclxuICB9IC8qIHJvdGF0ZVggKi9cclxuXHJcbiAgc3RhdGljIHJvdGF0ZVkoYW5nbGUpIHtcclxuICAgIGxldCBzID0gTWF0aC5zaW4oYW5nbGUpLCBjID0gTWF0aC5jb3MoYW5nbGUpO1xyXG5cclxuICAgIHJldHVybiBuZXcgTWF0NChcclxuICAgICAgYywgMCwtcywgMCxcclxuICAgICAgMCwgMSwgMCwgMCxcclxuICAgICAgcywgMCwgYywgMCxcclxuICAgICAgMCwgMCwgMCwgMVxyXG4gICAgKTtcclxuICB9IC8qIHJvdGF0ZVkgKi9cclxuXHJcbiAgc3RhdGljIHJvdGF0ZVooYW5nbGUpIHtcclxuICAgIGxldCBzID0gTWF0aC5zaW4oYW5nbGUpLCBjID0gTWF0aC5jb3MoYW5nbGUpO1xyXG5cclxuICAgIHJldHVybiBuZXcgTWF0NChcclxuICAgICAgYywgcywgMCwgMCxcclxuICAgICAtcywgYywgMCwgMCxcclxuICAgICAgMCwgMCwgMSwgMCxcclxuICAgICAgMCwgMCwgMCwgMVxyXG4gICAgKTtcclxuICB9IC8qIHJvdGF0ZVogKi9cclxuXHJcbiAgc3RhdGljIHZpZXcobG9jLCBhdCwgdXApIHtcclxuICAgIGxldFxyXG4gICAgICBkaXIgPSBhdC5zdWIobG9jKS5ub3JtYWxpemUoKSxcclxuICAgICAgcmdoID0gZGlyLmNyb3NzKHVwKS5ub3JtYWxpemUoKSxcclxuICAgICAgdHVwID0gcmdoLmNyb3NzKGRpcik7XHJcblxyXG4gICAgcmV0dXJuIG5ldyBNYXQ0KFxyXG4gICAgICByZ2gueCwgICAgICAgICB0dXAueCwgICAgICAgICAtZGlyLngsICAgICAgIDAsXHJcbiAgICAgIHJnaC55LCAgICAgICAgIHR1cC55LCAgICAgICAgIC1kaXIueSwgICAgICAgMCxcclxuICAgICAgcmdoLnosICAgICAgICAgdHVwLnosICAgICAgICAgLWRpci56LCAgICAgICAwLFxyXG4gICAgICAtbG9jLmRvdChyZ2gpLCAtbG9jLmRvdCh0dXApLCBsb2MuZG90KGRpciksIDFcclxuICAgICk7XHJcbiAgfSAvKiB2aWV3ICovXHJcblxyXG4gIHN0YXRpYyBmcnVzdHVtKGxlZnQsIHJpZ2h0LCBib3R0b20sIHRvcCwgbmVhciwgZmFyKSB7XHJcbiAgICByZXR1cm4gbmV3IE1hdDQoXHJcbiAgICAgIDIgKiBuZWFyIC8gKHJpZ2h0IC0gbGVmdCksICAgICAgIDAsICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgMCxcclxuICAgICAgMCwgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgMiAqIG5lYXIgLyAodG9wIC0gYm90dG9tKSwgICAgICAgMCwgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAwLFxyXG4gICAgICAocmlnaHQgKyBsZWZ0KSAvIChyaWdodCAtIGxlZnQpLCAodG9wICsgYm90dG9tKSAvICh0b3AgLSBib3R0b20pLCAobmVhciArIGZhcikgLyAobmVhciAtIGZhciksICAgLTEsXHJcbiAgICAgIDAsICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDAsICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDIgKiBuZWFyICogZmFyIC8gKG5lYXIgLSBmYXIpLCAgMFxyXG4gICAgKTtcclxuICB9IC8qIGZydXN0dW0gKi9cclxufSAvKiBNYXQ0ICovXHJcblxyXG5leHBvcnQgY2xhc3MgQ2FtZXJhIHtcclxuICAvLyBjYW1lcmEgcHJvamVjdGlvbiBzaGFwZSBwYXJhbXNcclxuICBwcm9qU2l6ZSA9IG5ldyBTaXplKDAuMDEsIDAuMDEpO1xyXG4gIG5lYXIgPSAwLjAxO1xyXG4gIGZhciA9IDgxOTI7XHJcbiAgXHJcbiAgLy8gY3VycmVudCBzY3JlZW4gcmVzb2x1dGlvblxyXG4gIHNjcmVlblNpemU7XHJcblxyXG4gIC8vIGNhbWVyYSBsb2NhdGlvblxyXG4gIGxvYztcclxuICBhdDtcclxuICBkaXI7XHJcbiAgdXA7XHJcbiAgcmlnaHQ7XHJcblxyXG4gIC8vIGNhbWVyYSBwcm9qZWN0aW9uIG1hdHJpY2VzXHJcbiAgdmlldztcclxuICBwcm9qO1xyXG4gIHZpZXdQcm9qO1xyXG5cclxuICBjb25zdHJ1Y3RvcigpIHtcclxuICAgIHRoaXMucHJvaiA9IE1hdDQuaWRlbnRpdHkoKTtcclxuICAgIHRoaXMuc2V0KG5ldyBWZWMzKDAsIDAsIC0xKSwgbmV3IFZlYzMoMCwgMCwgMCksIG5ldyBWZWMzKDAsIDEsIDApKTtcclxuICAgIHRoaXMucmVzaXplKG5ldyBTaXplKDMwLCAzMCkpO1xyXG4gIH0gLyogY29uc3RydWN0b3IgKi9cclxuXHJcbiAgcHJvalNldChuZXdOZWFyLCBuZXdGYXIsIG5ld1Byb2pTaXplID0gbmV3IFNpemUoMC4wMSwgMC4wMSkpIHtcclxuICAgIHRoaXMucHJvalNpemUgPSBuZXdQcm9qU2l6ZS5jb3B5KCk7XHJcbiAgICB0aGlzLm5lYXIgPSBuZXdOZWFyO1xyXG4gICAgdGhpcy5mYXIgPSBuZXdGYXI7XHJcblxyXG4gICAgaWYgKHRoaXMuc2NyZWVuU2l6ZS53ID4gdGhpcy5zY3JlZW5TaXplLmgpIHtcclxuICAgICAgdGhpcy5wcm9qU2l6ZS53ICo9IHRoaXMuc2NyZWVuU2l6ZS53IC8gdGhpcy5zY3JlZW5TaXplLmg7XHJcbiAgICB9IGVsc2Uge1xyXG4gICAgICB0aGlzLnByb2pTaXplLmggKj0gdGhpcy5zY3JlZW5TaXplLmggLyB0aGlzLnNjcmVlblNpemUudztcclxuICAgIH1cclxuXHJcbiAgICB0aGlzLnByb2ogPSBNYXQ0LmZydXN0dW0oLXRoaXMucHJvalNpemUudyAvIDIsIHRoaXMucHJvalNpemUudyAvIDIsIC10aGlzLnByb2pTaXplLmggLyAyLCB0aGlzLnByb2pTaXplLmggLyAyLCB0aGlzLm5lYXIsIHRoaXMuZmFyKTtcclxuICAgIHRoaXMudmlld1Byb2ogPSB0aGlzLnZpZXcubXVsKHRoaXMucHJvaik7XHJcbiAgfSAvKiBwcm9qU2V0ICovXHJcblxyXG4gIHJlc2l6ZShuZXdTY3JlZW5TaXplKSB7XHJcbiAgICB0aGlzLnNjcmVlblNpemUgPSBuZXdTY3JlZW5TaXplLmNvcHkoKTtcclxuICAgIHRoaXMucHJvalNldCh0aGlzLm5lYXIsIHRoaXMuZmFyKTtcclxuICB9IC8qIHJlc2l6ZSAqL1xyXG5cclxuICBzZXQobG9jLCBhdCwgdXApIHtcclxuICAgIHRoaXMudmlldyA9IE1hdDQudmlldyhsb2MsIGF0LCB1cCk7XHJcbiAgICB0aGlzLnZpZXdQcm9qID0gdGhpcy52aWV3Lm11bCh0aGlzLnByb2opO1xyXG5cclxuICAgIHRoaXMubG9jID0gbG9jLmNvcHkoKTtcclxuICAgIHRoaXMuYXQgPSBhdC5jb3B5KCk7XHJcblxyXG4gICAgdGhpcy5yaWdodCA9IG5ldyBWZWMzKHRoaXMudmlldy5tWyAwXSwgdGhpcy52aWV3Lm1bIDRdLCB0aGlzLnZpZXcubVsgOF0pO1xyXG4gICAgdGhpcy51cCAgICA9IG5ldyBWZWMzKHRoaXMudmlldy5tWyAxXSwgdGhpcy52aWV3Lm1bIDVdLCB0aGlzLnZpZXcubVsgOV0pO1xyXG4gICAgdGhpcy5kaXIgICA9IG5ldyBWZWMzKHRoaXMudmlldy5tWyAyXSwgdGhpcy52aWV3Lm1bIDZdLCB0aGlzLnZpZXcubVsxMF0pLm11bCgtMSk7XHJcbiAgfSAvKiBzZXQgKi9cclxufSAvKiBDYW1lcmEgKi8iLCJpbXBvcnQgKiBhcyBtdGggZnJvbSBcIi4vbXRoLmpzXCI7XHJcblxyXG4vKiBmb3JtYXQgZGVjb2RpbmcgZnVuY3Rpb24gKi9cclxuZnVuY3Rpb24gZ2V0Rm9ybWF0KGNvbXBvbmVudFR5cGUsIGNvbXBvbmVudENvdW50KSB7XHJcbiAgY29uc3QgZm10cyA9IFtXZWJHTDJSZW5kZXJpbmdDb250ZXh0LlJFRCwgV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5SRywgV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5SR0IsIFdlYkdMMlJlbmRlcmluZ0NvbnRleHQuUkdCQV07XHJcbiAgc3dpdGNoIChjb21wb25lbnRUeXBlKSB7XHJcbiAgICBjYXNlIFRleHR1cmUuRkxPQVQ6XHJcbiAgICAgIGNvbnN0IGZsb2F0SW50ZXJuYWxzID0gW1dlYkdMMlJlbmRlcmluZ0NvbnRleHQuUjMyRiwgV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5SRzMyRiwgV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5SR0IzMkYsIFdlYkdMMlJlbmRlcmluZ0NvbnRleHQuUkdCQTMyRl07XHJcbiAgICAgIHJldHVybiB7XHJcbiAgICAgICAgZm9ybWF0OiBmbXRzW2NvbXBvbmVudENvdW50IC0gMV0sXHJcbiAgICAgICAgaW50ZXJuYWw6IGZsb2F0SW50ZXJuYWxzW2NvbXBvbmVudENvdW50IC0gMV0sXHJcbiAgICAgICAgY29tcG9uZW50VHlwZTogV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5GTE9BVCxcclxuICAgICAgfTtcclxuXHJcbiAgICBjYXNlIFRleHR1cmUuVU5TSUdORURfQllURTpcclxuICAgICAgY29uc3QgYnl0ZUludGVybmFscyA9IFtXZWJHTDJSZW5kZXJpbmdDb250ZXh0LlI4LCBXZWJHTDJSZW5kZXJpbmdDb250ZXh0LlJHOCwgV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5SR0I4LCBXZWJHTDJSZW5kZXJpbmdDb250ZXh0LlJHQkE4XTtcclxuICAgICAgcmV0dXJuIHtcclxuICAgICAgICBmb3JtYXQ6IGZtdHNbY29tcG9uZW50Q291bnQgLSAxXSxcclxuICAgICAgICBpbnRlcm5hbDogYnl0ZUludGVybmFsc1tjb21wb25lbnRDb3VudCAtIDFdLFxyXG4gICAgICAgIGNvbXBvbmVudFR5cGU6IFdlYkdMMlJlbmRlcmluZ0NvbnRleHQuVU5TSUdORURfQllURSxcclxuICAgICAgfTtcclxuXHJcbiAgICBjYXNlIFRleHR1cmUuREVQVEg6XHJcbiAgICAgIHJldHVybiB7XHJcbiAgICAgICAgZm9ybWF0OiBXZWJHTDJSZW5kZXJpbmdDb250ZXh0LkRFUFRIX0NPTVBPTkVOVCxcclxuICAgICAgICBpbnRlcm5hbDogV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5ERVBUSF9DT01QT05FTlQzMkYsXHJcbiAgICAgICAgY29tcG9uZW50VHlwZTogV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5GTE9BVCxcclxuICAgICAgfTtcclxuXHJcbiAgICBkZWZhdWx0OlxyXG4gICAgICAvLyBtaW5pbWFsIGZvcm1hdCBwb3NzaWJsZVxyXG4gICAgICByZXR1cm4ge1xyXG4gICAgICAgIGZvcm1hdDogV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5SRUQsXHJcbiAgICAgICAgaW50ZXJuYWw6IFdlYkdMMlJlbmRlcmluZ0NvbnRleHQuUjgsXHJcbiAgICAgICAgY29tcG9uZW50VHlwZTogV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5VTlNJR05FRF9CWVRFLFxyXG4gICAgICB9O1xyXG4gIH1cclxufSAvKiBnZXRGb3JtYXQgKi9cclxuXHJcbmV4cG9ydCBjbGFzcyBUZXh0dXJlIHtcclxuICAjZ2w7XHJcbiAgI2Zvcm1hdDtcclxuICBzaXplO1xyXG4gIGlkO1xyXG5cclxuICBzdGF0aWMgRkxPQVQgICAgICAgICA9IDA7XHJcbiAgc3RhdGljIFVOU0lHTkVEX0JZVEUgPSAxO1xyXG4gIHN0YXRpYyBERVBUSCAgICAgICAgID0gMjtcclxuXHJcbiAgY29uc3RydWN0b3IoZ2wsIGNvbXBvbmVudFR5cGUgPSBUZXh0dXJlLkZMT0FULCBjb21wb25lbnRDb3VudCA9IDEpIHtcclxuICAgIHRoaXMuZ2wgPSBnbDtcclxuICAgIHRoaXMuc2l6ZSA9IG5ldyBtdGguU2l6ZSgxLCAxKTtcclxuICAgIHRoaXMuaWQgPSBnbC5jcmVhdGVUZXh0dXJlKCk7XHJcbiAgICBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0aGlzLmlkKTtcclxuXHJcbiAgICB0aGlzLmZvcm1hdCA9IGdldEZvcm1hdChjb21wb25lbnRUeXBlLCBjb21wb25lbnRDb3VudCk7XHJcbiAgICAvLyBwdXQgZW1wdHkgaW1hZ2UgZGF0YVxyXG4gICAgZ2wudGV4SW1hZ2UyRChnbC5URVhUVVJFXzJELCAwLCB0aGlzLmZvcm1hdC5pbnRlcm5hbCwgMSwgMSwgMCwgdGhpcy5mb3JtYXQuZm9ybWF0LCB0aGlzLmZvcm1hdC5jb21wb25lbnRUeXBlLCBudWxsKTtcclxuXHJcbiAgICBnbC50ZXhQYXJhbWV0ZXJpKGdsLlRFWFRVUkVfMkQsIGdsLlRFWFRVUkVfTUFHX0ZJTFRFUiwgZ2wuTElORUFSKTtcclxuICAgIGdsLnRleFBhcmFtZXRlcmkoZ2wuVEVYVFVSRV8yRCwgZ2wuVEVYVFVSRV9NSU5fRklMVEVSLCBnbC5MSU5FQVIpO1xyXG4gICAgZ2wudGV4UGFyYW1ldGVyaShnbC5URVhUVVJFXzJELCBnbC5URVhUVVJFX1dSQVBfUywgZ2wuUkVQRUFUKTtcclxuICAgIGdsLnRleFBhcmFtZXRlcmkoZ2wuVEVYVFVSRV8yRCwgZ2wuVEVYVFVSRV9XUkFQX1QsIGdsLlJFUEVBVCk7XHJcbiAgfSAvKiBjb25zdHJ1Y3RvciAqL1xyXG5cclxuICBcclxuICBzdGF0aWMgZGVmYXVsdENoZWNrZXJUZXh0dXJlID0gbnVsbDtcclxuICBzdGF0aWMgZGVmYXVsdENoZWNrZXIoZ2wpIHtcclxuICAgIGlmIChUZXh0dXJlLmRlZmF1bHRDaGVja2VyVGV4dHVyZSA9PT0gbnVsbCkge1xyXG4gICAgICBUZXh0dXJlLmRlZmF1bHRDaGVja2VyVGV4dHVyZSA9IG5ldyBUZXh0dXJlKGdsLCBUZXh0dXJlLlVOU0lHTkVEX0JZVEUsIDQpO1xyXG4gIFxyXG4gICAgICBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCBUZXh0dXJlLmRlZmF1bHRDaGVja2VyVGV4dHVyZS5pZCk7XHJcbiAgICAgIGdsLnRleEltYWdlMkQoZ2wuVEVYVFVSRV8yRCwgMCwgZ2wuUkdCQTgsIDEsIDEsIDAsIGdsLlJHQkEsIGdsLlVOU0lHTkVEX0JZVEUsIG5ldyBVaW50OEFycmF5KFsweDAwLCAweEZGLCAweDAwLCAweEZGXSkpO1xyXG5cclxuICAgICAgZ2wudGV4UGFyYW1ldGVyaShnbC5URVhUVVJFXzJELCBnbC5URVhUVVJFX01JTl9GSUxURVIsIGdsLk5FQVJFU1QpO1xyXG4gICAgICBnbC50ZXhQYXJhbWV0ZXJpKGdsLlRFWFRVUkVfMkQsIGdsLlRFWFRVUkVfTUFHX0ZJTFRFUiwgZ2wuTkVBUkVTVCk7XHJcbiAgICB9XHJcblxyXG4gICAgcmV0dXJuIFRleHR1cmUuZGVmYXVsdENoZWNrZXJUZXh0dXJlO1xyXG4gIH0gLyogZGVmYXVsdENoZWNrZXIgKi9cclxuXHJcbiAgZGVmYXVsdENoZWNrZXIoZGF0YSkge1xyXG4gICAgdGhpcy5nbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0aGlzLmlkKTtcclxuICAgIHRoaXMuZ2wudGV4SW1hZ2UyRChnbC5URVhUVVJFXzJELCAwLCBnbC5SR0JBLCAyLCAyLCAwLCBnbC5SR0JBLCBnbC5VTlNJR05FRF9CWVRFLCBuZXcgVWludDhBcnJheShbXHJcbiAgICAgIDB4MDAsIDB4RkYsIDB4MDAsIDB4RkYsXHJcbiAgICAgIDB4MDAsIDB4MDAsIDB4MDAsIDB4RkYsXHJcbiAgICAgIDB4MDAsIDB4MDAsIDB4MDAsIDB4RkYsXHJcbiAgICAgIDB4MDAsIDB4RkYsIDB4MDAsIDB4RkYsXHJcbiAgICBdKSk7XHJcbiAgfVxyXG5cclxuICBsb2FkKHBhdGgpIHtcclxuICAgIGxldCBnbCA9IHRoaXMuZ2w7XHJcblxyXG4gICAgLy8gYXN5bmMgaW1hZ2UgbG9hZGluZ1xyXG4gICAgbGV0IGltYWdlID0gbmV3IEltYWdlKCk7XHJcbiAgICBpbWFnZS5zcmMgPSBwYXRoO1xyXG4gICAgaW1hZ2UuYWRkRXZlbnRMaXN0ZW5lcihcImxvYWRcIiwgKCkgPT4ge1xyXG4gICAgICB0aGlzLnNpemUudyA9IGltYWdlLndpZHRoO1xyXG4gICAgICB0aGlzLnNpemUuaCA9IGltYWdlLmhlaWdodDtcclxuICAgICAgZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGhpcy5pZCk7XHJcbiAgICAgIGdsLnRleEltYWdlMkQoZ2wuVEVYVFVSRV8yRCwgMCwgdGhpcy5mb3JtYXQuaW50ZXJuYWwsIHRoaXMuZm9ybWF0LmZvcm1hdCwgdGhpcy5mb3JtYXQuY29tcG9uZW50VHlwZSwgaW1hZ2UpO1xyXG4gICAgfSk7XHJcbiAgfSAvKiBsb2FkICovXHJcblxyXG4gIGJpbmQocHJvZ3JhbSwgaW5kZXggPSAwKSB7XHJcbiAgICBsZXQgZ2wgPSB0aGlzLmdsO1xyXG5cclxuICAgIGdsLmFjdGl2ZVRleHR1cmUoZ2wuVEVYVFVSRTAgKyBpbmRleCk7XHJcbiAgICBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0aGlzLmlkKTtcclxuXHJcbiAgICBsZXQgbG9jYXRpb24gPSBnbC5nZXRVbmlmb3JtTG9jYXRpb24ocHJvZ3JhbSwgYFRleHR1cmUke2luZGV4fWApO1xyXG4gICAgZ2wudW5pZm9ybTFpKGxvY2F0aW9uLCBpbmRleCk7XHJcbiAgfSAvKiBiaW5kICovXHJcblxyXG4gIHJlc2l6ZShzaXplKSB7XHJcbiAgICBsZXQgZ2wgPSB0aGlzLmdsO1xyXG4gICAgdGhpcy5zaXplID0gc2l6ZS5jb3B5KCk7XHJcbiAgICBnbC50ZXhJbWFnZTJEKGdsLlRFWFRVUkVfMkQsIDAsIHRoaXMuZm9ybWF0LmludGVybmFsLCB0aGlzLnNpemUudywgdGhpcy5zaXplLmgsIDAsIHRoaXMuZm9ybWF0LmZvcm1hdCwgdGhpcy5mb3JtYXQuY29tcG9uZW50VHlwZSwgbnVsbCk7XHJcbiAgfSAvKiByZXNpemUgKi9cclxufSAvKiBUZXh0dXJlICovXHJcblxyXG5jb25zdCBmYWNlRGVzY3JpcHRpb25zID0gW1xyXG4gIHt0YXJnZXQ6IFdlYkdMMlJlbmRlcmluZ0NvbnRleHQuVEVYVFVSRV9DVUJFX01BUF9QT1NJVElWRV9YLCB0ZXh0OiBcIitYXCIsIHBhdGg6IFwicG9zWFwifSxcclxuICB7dGFyZ2V0OiBXZWJHTDJSZW5kZXJpbmdDb250ZXh0LlRFWFRVUkVfQ1VCRV9NQVBfTkVHQVRJVkVfWCwgdGV4dDogXCItWFwiLCBwYXRoOiBcIm5lZ1hcIn0sXHJcbiAge3RhcmdldDogV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5URVhUVVJFX0NVQkVfTUFQX1BPU0lUSVZFX1ksIHRleHQ6IFwiK1lcIiwgcGF0aDogXCJwb3NZXCJ9LFxyXG4gIHt0YXJnZXQ6IFdlYkdMMlJlbmRlcmluZ0NvbnRleHQuVEVYVFVSRV9DVUJFX01BUF9ORUdBVElWRV9ZLCB0ZXh0OiBcIi1ZXCIsIHBhdGg6IFwibmVnWVwifSxcclxuICB7dGFyZ2V0OiBXZWJHTDJSZW5kZXJpbmdDb250ZXh0LlRFWFRVUkVfQ1VCRV9NQVBfUE9TSVRJVkVfWiwgdGV4dDogXCIrWlwiLCBwYXRoOiBcInBvc1pcIn0sXHJcbiAge3RhcmdldDogV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5URVhUVVJFX0NVQkVfTUFQX05FR0FUSVZFX1osIHRleHQ6IFwiLVpcIiwgcGF0aDogXCJuZWdaXCJ9LFxyXG5dO1xyXG5cclxuZXhwb3J0IGNsYXNzIEN1YmVtYXAge1xyXG4gICNnbDtcclxuICBpZDtcclxuXHJcbiAgY29uc3RydWN0b3IoZ2wpIHtcclxuICAgIHRoaXMuZ2wgPSBnbDtcclxuXHJcbiAgICBsZXQgY3R4ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImNhbnZhc1wiKS5nZXRDb250ZXh0KFwiMmRcIik7XHJcblxyXG4gICAgY3R4LmNhbnZhcy53aWR0aCA9IDEyODtcclxuICAgIGN0eC5jYW52YXMuaGVpZ2h0ID0gMTI4O1xyXG5cclxuICAgIGZ1bmN0aW9uIGRyYXdGYWNlKHRleHQpIHtcclxuICAgICAgY29uc3Qge3dpZHRoLCBoZWlnaHR9ID0gY3R4LmNhbnZhcztcclxuXHJcbiAgICAgIGN0eC5maWxsU3R5bGUgPSAnI0NDQyc7XHJcbiAgICAgIGN0eC5maWxsUmVjdCgwLCAwLCB3aWR0aCwgaGVpZ2h0KTtcclxuICAgICAgY3R4LmZvbnQgPSBgJHt3aWR0aCAqIDAuNX1weCBjb25zb2xhc2A7XHJcbiAgICAgIGN0eC50ZXh0QWxpZ24gPSAnY2VudGVyJztcclxuICAgICAgY3R4LnRleHRCYXNlTGluZSA9ICdtaWRkbGUnO1xyXG4gICAgICBjdHguZmlsbFN0eWxlID0gJyMzMzMnO1xyXG5cclxuICAgICAgY3R4LmZpbGxUZXh0KHRleHQsIHdpZHRoIC8gMiwgaGVpZ2h0IC8gMik7XHJcbiAgICB9XHJcblxyXG4gICAgdGhpcy5pZCA9IGdsLmNyZWF0ZVRleHR1cmUoKTtcclxuICAgIGdsLmJpbmRUZXh0dXJlKGdsLlRFWFRVUkVfQ1VCRV9NQVAsIHRoaXMuaWQpO1xyXG5cclxuICAgIGZvciAobGV0IGRlc2NyIG9mIGZhY2VEZXNjcmlwdGlvbnMpIHtcclxuICAgICAgZHJhd0ZhY2UoZGVzY3IudGV4dCk7XHJcblxyXG4gICAgICBnbC50ZXhJbWFnZTJEKGRlc2NyLnRhcmdldCwgMCwgZ2wuUkdCQSwgZ2wuUkdCQSwgZ2wuVU5TSUdORURfQllURSwgY3R4LmNhbnZhcyk7XHJcbiAgICB9XHJcbiAgICBnbC5nZW5lcmF0ZU1pcG1hcChnbC5URVhUVVJFX0NVQkVfTUFQKTtcclxuICAgIGdsLnRleFBhcmFtZXRlcmkoZ2wuVEVYVFVSRV9DVUJFX01BUCwgZ2wuVEVYVFVSRV9NSU5fRklMVEVSLCBnbC5MSU5FQVJfTUlQTUFQX0xJTkVBUik7XHJcbiAgXHJcbiAgICBjdHguY2FudmFzLnJlbW92ZSgpO1xyXG4gIH0gLyogY29uc3RydWN0b3IgKi9cclxuXHJcbiAgbG9hZChwYXRoKSB7XHJcbiAgICBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFX0NVQkVfTUFQLCB0aGlzLmlkKTtcclxuXHJcbiAgICBmb3IgKGxldCBkZXNjciBvZiBmYWNlRGVzY3JpcHRpb25zKSB7XHJcbiAgICAgIGxldCBpbWFnZSA9IG5ldyBJbWFnZSgpO1xyXG5cclxuICAgICAgZ2wudGV4SW1hZ2UyRChkZXNjci50YXJnZXQsIDAsIGdsLlJHQkEsIGdsLlJHQkEsIGdsLlVOU0lHTkVEX0JZVEUsIGN0eC5jYW52YXMpO1xyXG4gICAgfVxyXG4gICAgZ2wuZ2VuZXJhdGVNaXBtYXAoZ2wuVEVYVFVSRV9DVUJFX01BUCk7XHJcbiAgICBnbC50ZXhQYXJhbWV0ZXJpKGdsLlRFWFRVUkVfQ1VCRV9NQVAsIGdsLlRFWFRVUkVfTUlOX0ZJTFRFUiwgZ2wuTElORUFSX01JUE1BUF9MSU5FQVIpO1xyXG4gIH0gLyogbG9hZCAqL1xyXG5cclxuICBiaW5kKHByb2dyYW0sIGluZGV4ID0gMCkge1xyXG4gICAgbGV0IGdsID0gdGhpcy5nbDtcclxuXHJcbiAgICBnbC5hY3RpdmVUZXh0dXJlKGdsLlRFWFRVUkUwICsgaW5kZXgpO1xyXG4gICAgZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV9DVUJFX01BUCwgdGhpcy5pZCk7XHJcblxyXG4gICAgbGV0IGxvY2F0aW9uID0gZ2wuZ2V0VW5pZm9ybUxvY2F0aW9uKHByb2dyYW0sIGBUZXh0dXJlJHtpbmRleH1gKTtcclxuICAgIGdsLnVuaWZvcm0xaShsb2NhdGlvbiwgaW5kZXgpO1xyXG4gIH0gLyogYmluZCAqL1xyXG59IC8qIEN1YmVtYXAgKi8iLCJleHBvcnQgY2xhc3MgVUJPIHtcclxuICBnbDtcclxuICBidWZmZXI7XHJcbiAgaXNFbXB0eSA9IHRydWU7XHJcblxyXG4gIGNvbnN0cnVjdG9yKGdsKSB7XHJcbiAgICB0aGlzLmdsID0gZ2w7XHJcbiAgICB0aGlzLmJ1ZmZlciA9IGdsLmNyZWF0ZUJ1ZmZlcigpO1xyXG4gIH0gLyogY29uc3RydWN0b3IgKi9cclxuXHJcbiAgd3JpdGVEYXRhKGRhdGFBc0Zsb2F0QXJyYXkpIHtcclxuICAgIHRoaXMuaXNFbXB0eSA9IGZhbHNlO1xyXG4gICAgdGhpcy5nbC5iaW5kQnVmZmVyKHRoaXMuZ2wuVU5JRk9STV9CVUZGRVIsIHRoaXMuYnVmZmVyKTtcclxuICAgIHRoaXMuZ2wuYnVmZmVyRGF0YSh0aGlzLmdsLlVOSUZPUk1fQlVGRkVSLCBkYXRhQXNGbG9hdEFycmF5LCB0aGlzLmdsLlNUQVRJQ19EUkFXKTtcclxuICB9IC8qIHdyaXRlRGF0YSAqL1xyXG5cclxuICBiaW5kKHNoYWRlciwgYmluZGluZ1BvaW50LCBidWZmZXJOYW1lKSB7XHJcbiAgICBpZiAoIXRoaXMuaXNFbXB0eSkge1xyXG4gICAgICBsZXQgbG9jYXRpb24gPSB0aGlzLmdsLmdldFVuaWZvcm1CbG9ja0luZGV4KHNoYWRlciwgYnVmZmVyTmFtZSk7XHJcblxyXG4gICAgICBpZiAobG9jYXRpb24gIT0gLTEpIHtcclxuICAgICAgICB0aGlzLmdsLnVuaWZvcm1CbG9ja0JpbmRpbmcoc2hhZGVyLCBsb2NhdGlvbiwgYmluZGluZ1BvaW50KTtcclxuICAgICAgICB0aGlzLmdsLmJpbmRCdWZmZXJCYXNlKHRoaXMuZ2wuVU5JRk9STV9CVUZGRVIsIGJpbmRpbmdQb2ludCwgdGhpcy5idWZmZXIpO1xyXG4gICAgICB9XHJcbiAgICB9XHJcbiAgfSAvKiBiaW5kICovXHJcbn0gLyogVUJPICovIiwiaW1wb3J0IHtsb2FkU2hhZGVyfSBmcm9tIFwiLi9zaGFkZXIuanNcIjtcclxuaW1wb3J0IHtUZXh0dXJlLCBDdWJlbWFwfSBmcm9tIFwiLi90ZXh0dXJlLmpzXCI7XHJcbmltcG9ydCB7VUJPfSBmcm9tIFwiLi91Ym8uanNcIjtcclxuZXhwb3J0IHtUZXh0dXJlLCBDdWJlbWFwLCBVQk8sIGxvYWRTaGFkZXJ9O1xyXG5cclxuZXhwb3J0IGNsYXNzIE1hdGVyaWFsIHtcclxuICB1Ym9OYW1lT25TaGFkZXIgPSBcIlwiO1xyXG4gIGdsO1xyXG4gIHVibyA9IG51bGw7ICAgIC8vIG9iamVjdCBidWZmZXJcclxuICB0ZXh0dXJlcyA9IFtdOyAvLyBhcnJheSBvZiB0ZXh0dXJlc1xyXG4gIHNoYWRlcjsgICAgICAgIC8vIHNoYWRlciBwb2ludGVyXHJcblxyXG4gIGNvbnN0cnVjdG9yKGdsLCBzaGFkZXIpIHtcclxuICAgIHRoaXMuZ2wgPSBnbDtcclxuICAgIHRoaXMuc2hhZGVyID0gc2hhZGVyO1xyXG4gIH0gLyogY29uc3RydWN0b3IgKi9cclxuXHJcbiAgYXBwbHkoKSB7XHJcbiAgICBsZXQgZ2wgPSB0aGlzLmdsO1xyXG5cclxuICAgIGdsLnVzZVByb2dyYW0odGhpcy5zaGFkZXIpO1xyXG5cclxuICAgIGlmICh0aGlzLnVibyAhPSBudWxsKVxyXG4gICAgICB0aGlzLnViby5iaW5kKHRoaXMuc2hhZGVyLCAwLCB0aGlzLnVib05hbWVPblNoYWRlcik7XHJcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMudGV4dHVyZXMubGVuZ3RoOyBpKyspXHJcbiAgICAgIHRoaXMudGV4dHVyZXNbaV0uYmluZCh0aGlzLnNoYWRlciwgaSk7XHJcbiAgfSAvKiBhcHBseSAqL1xyXG59IC8qIE1hdGVyaWFsICovIiwiaW1wb3J0IHtsb2FkU2hhZGVyfSBmcm9tIFwiLi9zaGFkZXIuanNcIjtcclxuaW1wb3J0IHtNYXRlcmlhbCwgVGV4dHVyZSwgQ3ViZW1hcCwgVUJPfSBmcm9tIFwiLi9tYXRlcmlhbC5qc1wiO1xyXG5pbXBvcnQgKiBhcyBtdGggZnJvbSBcIi4vbXRoLmpzXCI7XHJcblxyXG5leHBvcnQge2xvYWRTaGFkZXIsIE1hdGVyaWFsLCBUZXh0dXJlLCBDdWJlbWFwLCBVQk8sIG10aH07XHJcblxyXG5leHBvcnQgY2xhc3MgVmVydGV4IHtcclxuICBwO1xyXG4gIHQ7XHJcbiAgbjtcclxuXHJcbiAgY29uc3RydWN0b3IocG9zaXRpb24sIHRleGNvb3JkLCBub3JtYWwpIHtcclxuICAgIHRoaXMucCA9IHBvc2l0aW9uO1xyXG4gICAgdGhpcy50ID0gdGV4Y29vcmQ7XHJcbiAgICB0aGlzLm4gPSBub3JtYWw7XHJcbiAgfSAvKiBjb25zdHJ1Y3RvciAqL1xyXG5cclxuICBzdGF0aWMgZnJvbUNvb3JkKHB4LCBweSwgcHosIHB1ID0gMCwgcHYgPSAwLCBwbnggPSAwLCBwbnkgPSAwLCBwbnogPSAxKSB7XHJcbiAgICByZXR1cm4gbmV3IFZlcnRleChuZXcgbXRoLlZlYzMocHgsIHB5LCBweiksIG5ldyBtdGguVmVjMihwdSwgcHYpLCBuZXcgbXRoLlZlYzMocG54LCBwbnksIHBueikpO1xyXG4gIH0gLyogZnJvbUNvb3JkICovXHJcblxyXG4gIHN0YXRpYyBmcm9tVmVjdG9ycyhwLCB0ID0gbmV3IG10aC5WZWMyKDAsIDApLCBuID0gbmV3IG10aC5WZWMzKDEsIDEsIDEpKSB7XHJcbiAgICByZXR1cm4gbmV3IFZlcnRleChwLmNvcHkoKSwgdC5jb3B5KCksIG4uY29weSgpKTtcclxuICB9IC8qIGZyb21WZWN0b3JzICovXHJcbn07IC8qIFZlcnRleCAqL1xyXG5cclxuZXhwb3J0IGNsYXNzIFRvcG9sb2d5IHtcclxuICB2dHg7XHJcbiAgaWR4O1xyXG4gIHR5cGUgPSBUb3BvbG9neS5UUklBTkdMRVM7XHJcblxyXG4gIHN0YXRpYyBMSU5FUyAgICAgICAgICA9IFdlYkdMMlJlbmRlcmluZ0NvbnRleHQuTElORVM7XHJcbiAgc3RhdGljIExJTkVfU1RSSVAgICAgID0gV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5MSU5FX1NUUklQO1xyXG4gIHN0YXRpYyBMSU5FX0xPT1AgICAgICA9IFdlYkdMMlJlbmRlcmluZ0NvbnRleHQuTElORV9MT09QO1xyXG5cclxuICBzdGF0aWMgUE9JTlRTICAgICAgICAgPSBXZWJHTDJSZW5kZXJpbmdDb250ZXh0LlBPSU5UUztcclxuXHJcbiAgc3RhdGljIFRSSUFOR0xFUyAgICAgID0gV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5UUklBTkdMRVM7XHJcbiAgc3RhdGljIFRSSUFOR0xFX1NUUklQID0gV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5UUklBTkdMRV9TVFJJUDtcclxuICBzdGF0aWMgVFJJQU5HTEVfRkFOICAgPSBXZWJHTDJSZW5kZXJpbmdDb250ZXh0LlRSSUFOR0xFX0ZBTjtcclxuXHJcbiAgY29uc3RydWN0b3IobnZ0eCA9IFtdLCBuaWR4ID0gbnVsbCkge1xyXG4gICAgdGhpcy52dHggPSBudnR4O1xyXG4gICAgdGhpcy5pZHggPSBuaWR4O1xyXG4gIH0gLyogY29uc3RydWN0b3IgKi9cclxuXHJcbiAgc3RhdGljIGdlb21ldHJ5VHlwZVRvR0woZ2VvbWV0cnlUeXBlKSB7XHJcbiAgICByZXR1cm4gZ2VvbWV0cnlUeXBlO1xyXG4gIH0gLyogZ2VvbWV0cnlUeXBlVG9HTCAqL1xyXG5cclxuICBzdGF0aWMgc3F1YXJlKCkge1xyXG4gICAgbGV0IHRwbCA9IG5ldyBUb3BvbG9neShbXHJcbiAgICAgIFZlcnRleC5mcm9tQ29vcmQoLTEsIC0xLCAwLCAwLCAwKSxcclxuICAgICAgVmVydGV4LmZyb21Db29yZCgtMSwgIDEsIDAsIDAsIDEpLFxyXG4gICAgICBWZXJ0ZXguZnJvbUNvb3JkKCAxLCAtMSwgMCwgMSwgMCksXHJcbiAgICAgIFZlcnRleC5mcm9tQ29vcmQoIDEsICAxLCAwLCAxLCAxKVxyXG4gICAgXSwgWzAsIDEsIDIsIDNdKTtcclxuICAgIHRwbC50eXBlID0gVG9wb2xvZ3kuVFJJQU5HTEVfU1RSSVA7XHJcbiAgICByZXR1cm4gdHBsO1xyXG4gIH0gLyogdGhlVHJpYW5nbGUgKi9cclxuXHJcbiAgc3RhdGljICNwbGFuZUluZGV4ZWQod2lkdGggPSAzMCwgaGVpZ2h0ID0gMzApIHtcclxuICAgIGxldCB0cGwgPSBuZXcgVG9wb2xvZ3koKTtcclxuXHJcbiAgICB0cGwudHlwZSA9IFRvcG9sb2d5LlRSSUFOR0xFX1NUUklQO1xyXG4gICAgdHBsLnZ0eCA9IFtdO1xyXG4gICAgdHBsLmlkeCA9IFtdO1xyXG5cclxuICAgIGxldCBpID0gMDtcclxuICAgIGZvciAobGV0IHkgPSAwOyB5IDwgaGVpZ2h0IC0gMTsgeSsrKSB7XHJcbiAgICAgIGZvciAobGV0IHggPSAwOyB4IDwgd2lkdGg7IHgrKykge1xyXG4gICAgICAgIHRwbC5pZHhbaSsrXSA9IHkgKiB3aWR0aCArIHg7XHJcbiAgICAgICAgdHBsLmlkeFtpKytdID0gKHkgKyAxKSAqIHdpZHRoICsgeDtcclxuICAgICAgfVxyXG4gICAgICB0cGwuaWR4W2krK10gPSAtMTtcclxuICAgIH1cclxuXHJcbiAgICByZXR1cm4gdHBsO1xyXG4gIH0gLyogcGxhbmVJbmRleGVkICovXHJcblxyXG4gIHN0YXRpYyBzcGhlcmUocmFkaXVzID0gMSwgd2lkdGggPSAzMCwgaGVpZ2h0ID0gMzApIHtcclxuICAgIGxldCB0cGwgPSBUb3BvbG9neS4jcGxhbmVJbmRleGVkKHdpZHRoLCBoZWlnaHQpO1xyXG5cclxuICAgIGZvciAobGV0IHkgPSAwOyB5IDwgaGVpZ2h0OyB5KyspIHtcclxuICAgICAgbGV0IHRoZXRhID0gTWF0aC5QSSAqIHkgLyAoaGVpZ2h0IC0gMSk7XHJcbiAgICAgIGxldCBzdGhldGEgPSBNYXRoLnNpbih0aGV0YSk7XHJcbiAgICAgIGxldCBjdGhldGEgPSBNYXRoLmNvcyh0aGV0YSk7XHJcblxyXG4gICAgICBmb3IgKGxldCB4ID0gMDsgeCA8IHdpZHRoOyB4KyspIHtcclxuICAgICAgICBsZXQgcGhpID0gMiAqIE1hdGguUEkgKiB4IC8gKHdpZHRoIC0gMSk7XHJcblxyXG4gICAgICAgIGxldCBueCA9IHN0aGV0YSAqIE1hdGguc2luKHBoaSk7XHJcbiAgICAgICAgbGV0IG55ID0gY3RoZXRhO1xyXG4gICAgICAgIGxldCBueiA9IHN0aGV0YSAqIE1hdGguY29zKHBoaSk7XHJcblxyXG4gICAgICAgIHRwbC52dHhbeSAqIHdpZHRoICsgeF0gPSBWZXJ0ZXguZnJvbUNvb3JkKFxyXG4gICAgICAgICAgcmFkaXVzICogbngsIHJhZGl1cyAqIG55LCByYWRpdXMgKiBueixcclxuICAgICAgICAgIHggLyAod2lkdGggLSAxKSwgeSAvIChoZWlnaHQgLSAxKSxcclxuICAgICAgICAgIG54LCBueSwgbnpcclxuICAgICAgICApO1xyXG4gICAgICB9XHJcbiAgICB9XHJcblxyXG4gICAgcmV0dXJuIHRwbDtcclxuICB9IC8qIHNwaGVyZSAqL1xyXG5cclxuICBzdGF0aWMgYXN5bmMgbW9kZWxfb2JqKHBhdGgpIHtcclxuICAgIGxldCB0cGwgPSBuZXcgVG9wb2xvZ3koKTtcclxuICAgIHRwbC52dHggPSBbXTtcclxuICAgIHRwbC50eXBlID0gVG9wb2xvZ3kuVFJJQU5HTEVTO1xyXG5cclxuICAgIGNvbnN0IHNyYyA9IGF3YWl0IGZldGNoKHBhdGgpLnRoZW4ocmVzcG9uc2UgPT4gcmVzcG9uc2UudGV4dCgpKTtcclxuICAgIGxldCBsaW5lcyA9IHNyYy5zcGxpdChcIlxcblwiKTtcclxuICAgIGxldCBwb3NpdGlvbnMgPSBbXTtcclxuICAgIGxldCB0ZXhDb29yZHMgPSBbXTtcclxuICAgIGxldCBub3JtYWxzID0gW107XHJcblxyXG4gICAgZm9yIChsZXQgbGkgPSAwLCBsaW5lQ291bnQgPSBsaW5lcy5sZW5ndGg7IGxpIDwgbGluZUNvdW50OyBsaSsrKSB7XHJcbiAgICAgIGxldCBzZWdtZW50cyA9IGxpbmVzW2xpXS5zcGxpdChcIiBcIik7XHJcblxyXG4gICAgICBzd2l0Y2ggKHNlZ21lbnRzWzBdKSB7XHJcbiAgICAgICAgY2FzZSBcInZcIjpcclxuICAgICAgICAgIHBvc2l0aW9ucy5wdXNoKG5ldyBtdGguVmVjMyhcclxuICAgICAgICAgICAgcGFyc2VGbG9hdChzZWdtZW50c1sxXSksXHJcbiAgICAgICAgICAgIHBhcnNlRmxvYXQoc2VnbWVudHNbMl0pLFxyXG4gICAgICAgICAgICBwYXJzZUZsb2F0KHNlZ21lbnRzWzNdKVxyXG4gICAgICAgICAgKSk7XHJcbiAgICAgICAgICBicmVhaztcclxuXHJcbiAgICAgICAgY2FzZSBcInZ0XCI6XHJcbiAgICAgICAgICB0ZXhDb29yZHMucHVzaChuZXcgbXRoLlZlYzIoXHJcbiAgICAgICAgICAgIHBhcnNlRmxvYXQoc2VnbWVudHNbMV0pLFxyXG4gICAgICAgICAgICBwYXJzZUZsb2F0KHNlZ21lbnRzWzJdKVxyXG4gICAgICAgICAgKSk7XHJcbiAgICAgICAgICBicmVhaztcclxuXHJcbiAgICAgICAgY2FzZSBcInZuXCI6XHJcbiAgICAgICAgICBub3JtYWxzLnB1c2gobmV3IG10aC5WZWMzKFxyXG4gICAgICAgICAgICBwYXJzZUZsb2F0KHNlZ21lbnRzWzFdKSxcclxuICAgICAgICAgICAgcGFyc2VGbG9hdChzZWdtZW50c1syXSksXHJcbiAgICAgICAgICAgIHBhcnNlRmxvYXQoc2VnbWVudHNbM10pXHJcbiAgICAgICAgICApKTtcclxuICAgICAgICAgIGJyZWFrO1xyXG5cclxuICAgICAgICBjYXNlIFwiZlwiOlxyXG4gICAgICAgICAge1xyXG4gICAgICAgICAgICBsZXQgdnRkID0gc2VnbWVudHNbMV0uc3BsaXQoXCIvXCIpO1xyXG4gICAgICAgICAgICBsZXQgaTAgPSBwYXJzZUludCh2dGRbMF0pLCBpMSA9IHBhcnNlSW50KHZ0ZFsxXSksIGkyID0gcGFyc2VJbnQodnRkWzJdKTtcclxuXHJcbiAgICAgICAgICAgIHRwbC52dHgucHVzaChuZXcgVmVydGV4KFxyXG4gICAgICAgICAgICAgIE51bWJlci5pc05hTihpMCkgPyBuZXcgbXRoLlZlYzMoMCwgMCwgMCkgOiBwb3NpdGlvbnNbaTAgLSAxXSxcclxuICAgICAgICAgICAgICBOdW1iZXIuaXNOYU4oaTEpID8gbmV3IG10aC5WZWMyKDAsIDApIDogdGV4Q29vcmRzW2kxIC0gMV0sXHJcbiAgICAgICAgICAgICAgTnVtYmVyLmlzTmFOKGkyKSA/IG5ldyBtdGguVmVjMygwLCAwLCAwKSA6IG5vcm1hbHNbaTIgLSAxXVxyXG4gICAgICAgICAgICApKTtcclxuICAgICAgICAgIH1cclxuICAgICAgICAgIHtcclxuICAgICAgICAgICAgbGV0IHZ0ZCA9IHNlZ21lbnRzWzJdLnNwbGl0KFwiL1wiKTtcclxuICAgICAgICAgICAgbGV0IGkwID0gcGFyc2VJbnQodnRkWzBdKSwgaTEgPSBwYXJzZUludCh2dGRbMV0pLCBpMiA9IHBhcnNlSW50KHZ0ZFsyXSk7XHJcblxyXG4gICAgICAgICAgICB0cGwudnR4LnB1c2gobmV3IFZlcnRleChcclxuICAgICAgICAgICAgICBOdW1iZXIuaXNOYU4oaTApID8gbmV3IG10aC5WZWMzKDAsIDAsIDApIDogcG9zaXRpb25zW2kwIC0gMV0sXHJcbiAgICAgICAgICAgICAgTnVtYmVyLmlzTmFOKGkxKSA/IG5ldyBtdGguVmVjMigwLCAwKSA6IHRleENvb3Jkc1tpMSAtIDFdLFxyXG4gICAgICAgICAgICAgIE51bWJlci5pc05hTihpMikgPyBuZXcgbXRoLlZlYzMoMCwgMCwgMCkgOiBub3JtYWxzW2kyIC0gMV1cclxuICAgICAgICAgICAgKSk7XHJcbiAgICAgICAgICB9XHJcbiAgICAgICAgICB7XHJcbiAgICAgICAgICAgIGxldCB2dGQgPSBzZWdtZW50c1szXS5zcGxpdChcIi9cIik7XHJcbiAgICAgICAgICAgIGxldCBpMCA9IHBhcnNlSW50KHZ0ZFswXSksIGkxID0gcGFyc2VJbnQodnRkWzFdKSwgaTIgPSBwYXJzZUludCh2dGRbMl0pO1xyXG5cclxuICAgICAgICAgICAgdHBsLnZ0eC5wdXNoKG5ldyBWZXJ0ZXgoXHJcbiAgICAgICAgICAgICAgTnVtYmVyLmlzTmFOKGkwKSA/IG5ldyBtdGguVmVjMygwLCAwLCAwKSA6IHBvc2l0aW9uc1tpMCAtIDFdLFxyXG4gICAgICAgICAgICAgIE51bWJlci5pc05hTihpMSkgPyBuZXcgbXRoLlZlYzIoMCwgMCkgOiB0ZXhDb29yZHNbaTEgLSAxXSxcclxuICAgICAgICAgICAgICBOdW1iZXIuaXNOYU4oaTIpID8gbmV3IG10aC5WZWMzKDAsIDAsIDApIDogbm9ybWFsc1tpMiAtIDFdXHJcbiAgICAgICAgICAgICkpO1xyXG4gICAgICAgICAgfVxyXG4gICAgICAgIGJyZWFrO1xyXG4gICAgICB9XHJcbiAgICB9XHJcblxyXG4gICAgcmV0dXJuIHRwbDtcclxuICB9IC8qIG1vZGVsX29iaiAqL1xyXG5cclxuICBnZXRWZXJ0aWNlc0FzRmxvYXRBcnJheSgpIHtcclxuICAgIGxldCByZXNfYXJyYXkgPSBuZXcgRmxvYXQzMkFycmF5KHRoaXMudnR4Lmxlbmd0aCAqIDgpO1xyXG4gICAgbGV0IGkgPSAwO1xyXG4gICAgbGV0IG1pID0gdGhpcy52dHgubGVuZ3RoICogODtcclxuXHJcbiAgICB3aGlsZShpIDwgbWkpIHtcclxuICAgICAgbGV0IHZ0ID0gdGhpcy52dHhbaSA+PiAzXTtcclxuICBcclxuICAgICAgcmVzX2FycmF5W2krK10gPSB2dC5wLng7XHJcbiAgICAgIHJlc19hcnJheVtpKytdID0gdnQucC55O1xyXG4gICAgICByZXNfYXJyYXlbaSsrXSA9IHZ0LnAuejtcclxuXHJcbiAgICAgIHJlc19hcnJheVtpKytdID0gdnQudC54O1xyXG4gICAgICByZXNfYXJyYXlbaSsrXSA9IHZ0LnQueTtcclxuXHJcbiAgICAgIHJlc19hcnJheVtpKytdID0gdnQubi54O1xyXG4gICAgICByZXNfYXJyYXlbaSsrXSA9IHZ0Lm4ueTtcclxuICAgICAgcmVzX2FycmF5W2krK10gPSB2dC5uLno7XHJcbiAgICB9XHJcblxyXG4gICAgcmV0dXJuIHJlc19hcnJheTtcclxuICB9IC8qIGdldFZlcnRpY2VzQXNGbG9hdEFycmF5ICovXHJcblxyXG4gIGdldEluZGljZXNBc1VpbnRBcnJheSgpIHtcclxuICAgIHJldHVybiBuZXcgVWludDMyQXJyYXkodGhpcy5pZHgpO1xyXG4gIH0gLyogZ2V0SW5kaWNlc0FzVWludEFycmF5ICovXHJcbn0gLyogVG9wb2xvZ3kgKi9cclxuXHJcbmV4cG9ydCBjbGFzcyBQcmltaXRpdmUge1xyXG4gIGdsO1xyXG4gIHZlcnRleEFycmF5T2JqZWN0ID0gbnVsbDtcclxuICBpbmRleEJ1ZmZlciA9IG51bGw7XHJcbiAgdmVydGV4QnVmZmVyID0gbnVsbDtcclxuICB2ZXJ0ZXhOdW1iZXIgPSAwO1xyXG4gIGluZGV4TnVtYmVyID0gMDtcclxuICBnZW9tZXRyeVR5cGUgPSBUb3BvbG9neS5UUklBTkdMRVM7XHJcbiAgbWF0ZXJpYWwgPSBudWxsO1xyXG5cclxuICBjb25zdHJ1Y3RvcihnbENvbnRleHQpIHtcclxuICAgIHRoaXMuZ2wgPSBnbENvbnRleHQ7XHJcbiAgfSAvKiBjb25zdHJ1Y3RvciAqL1xyXG5cclxuICBkcmF3KGNhbWVyYUJ1ZmZlciA9IG51bGwpIHtcclxuICAgIGxldCBnbCA9IHRoaXMuZ2w7XHJcblxyXG4gICAgdGhpcy5tYXRlcmlhbC5hcHBseSgpO1xyXG5cclxuICAgIGlmIChjYW1lcmFCdWZmZXIgIT0gbnVsbCkge1xyXG4gICAgICBjYW1lcmFCdWZmZXIuYmluZCh0aGlzLm1hdGVyaWFsLnNoYWRlciwgMSwgXCJjYW1lcmFCdWZmZXJcIik7XHJcbiAgICB9XHJcblxyXG4gICAgZ2wuYmluZFZlcnRleEFycmF5KHRoaXMudmVydGV4QXJyYXlPYmplY3QpO1xyXG4gICAgaWYgKHRoaXMuaW5kZXhCdWZmZXIgIT0gbnVsbCkge1xyXG4gICAgICBnbC5iaW5kQnVmZmVyKGdsLkVMRU1FTlRfQVJSQVlfQlVGRkVSLCB0aGlzLmluZGV4QnVmZmVyKTtcclxuICAgICAgZ2wuZHJhd0VsZW1lbnRzKFRvcG9sb2d5Lmdlb21ldHJ5VHlwZVRvR0wodGhpcy5nZW9tZXRyeVR5cGUpLCB0aGlzLmluZGV4TnVtYmVyLCBnbC5VTlNJR05FRF9JTlQsIDApO1xyXG4gICAgfSBlbHNlIHtcclxuICAgICAgZ2wuZHJhd0FycmF5cyhUb3BvbG9neS5nZW9tZXRyeVR5cGVUb0dMKHRoaXMuZ2VvbWV0cnlUeXBlKSwgMCwgdGhpcy52ZXJ0ZXhOdW1iZXIpO1xyXG4gICAgfVxyXG4gIH0gLyogZHJhdyAqL1xyXG5cclxuICBzdGF0aWMgYXN5bmMgZnJvbVRvcG9sb2d5KGdsLCB0cGwsIG1hdGVyaWFsKSB7XHJcbiAgICBsZXQgcHJpbSA9IG5ldyBQcmltaXRpdmUoZ2wpO1xyXG4gICAgcHJpbS5tYXRlcmlhbCA9IG1hdGVyaWFsO1xyXG5cclxuICAgIHByaW0uZ2VvbWV0cnlUeXBlID0gdHBsLnR5cGU7XHJcblxyXG4gICAgLy8gQ3JlYXRlIHZlcnRleCBhcnJheVxyXG4gICAgcHJpbS52ZXJ0ZXhBcnJheU9iamVjdCA9IGdsLmNyZWF0ZVZlcnRleEFycmF5KCk7XHJcbiAgICBnbC5iaW5kVmVydGV4QXJyYXkocHJpbS52ZXJ0ZXhBcnJheU9iamVjdCk7XHJcblxyXG4gICAgLy8gV3JpdGUgdmVydGV4IGJ1ZmZlclxyXG4gICAgcHJpbS52ZXJ0ZXhCdWZmZXIgPSBnbC5jcmVhdGVCdWZmZXIoKTtcclxuICAgIGdsLmJpbmRCdWZmZXIoZ2wuQVJSQVlfQlVGRkVSLCBwcmltLnZlcnRleEJ1ZmZlcik7XHJcbiAgICBnbC5idWZmZXJEYXRhKGdsLkFSUkFZX0JVRkZFUiwgdHBsLmdldFZlcnRpY2VzQXNGbG9hdEFycmF5KCksIGdsLlNUQVRJQ19EUkFXKTtcclxuICAgIHByaW0udmVydGV4TnVtYmVyID0gdHBsLnZ0eC5sZW5ndGg7XHJcblxyXG4gICAgLy8gTWFwIHZlcnRleCBsYXlvdXRcclxuICAgIGxldCBwb3NpdGlvbkxvY2F0aW9uID0gZ2wuZ2V0QXR0cmliTG9jYXRpb24ocHJpbS5tYXRlcmlhbC5zaGFkZXIsIFwiaW5Qb3NpdGlvblwiKTtcclxuICAgIGlmIChwb3NpdGlvbkxvY2F0aW9uICE9IC0xKSB7XHJcbiAgICAgIGdsLnZlcnRleEF0dHJpYlBvaW50ZXIocG9zaXRpb25Mb2NhdGlvbiwgMywgZ2wuRkxPQVQsIGZhbHNlLCA4ICogNCwgMCk7XHJcbiAgICAgIGdsLmVuYWJsZVZlcnRleEF0dHJpYkFycmF5KHBvc2l0aW9uTG9jYXRpb24pO1xyXG4gICAgfVxyXG5cclxuICAgIGxldCB0ZXhDb29yZExvY2F0aW9uID0gZ2wuZ2V0QXR0cmliTG9jYXRpb24ocHJpbS5tYXRlcmlhbC5zaGFkZXIsIFwiaW5UZXhDb29yZFwiKTtcclxuICAgIGlmICh0ZXhDb29yZExvY2F0aW9uICE9IC0xKSB7XHJcbiAgICAgIGdsLnZlcnRleEF0dHJpYlBvaW50ZXIodGV4Q29vcmRMb2NhdGlvbiwgMywgZ2wuRkxPQVQsIGZhbHNlLCA4ICogNCwgMyAqIDQpO1xyXG4gICAgICBnbC5lbmFibGVWZXJ0ZXhBdHRyaWJBcnJheSh0ZXhDb29yZExvY2F0aW9uKTtcclxuICAgIH1cclxuXHJcbiAgICBsZXQgbm9ybWFsTG9jYXRpb24gPSBnbC5nZXRBdHRyaWJMb2NhdGlvbihwcmltLm1hdGVyaWFsLnNoYWRlciwgXCJpbk5vcm1hbFwiKTtcclxuICAgIGlmIChub3JtYWxMb2NhdGlvbiAhPSAtMSkge1xyXG4gICAgICBnbC52ZXJ0ZXhBdHRyaWJQb2ludGVyKG5vcm1hbExvY2F0aW9uLCAzLCBnbC5GTE9BVCwgZmFsc2UsIDggKiA0LCA1ICogNCk7XHJcbiAgICAgIGdsLmVuYWJsZVZlcnRleEF0dHJpYkFycmF5KG5vcm1hbExvY2F0aW9uKTtcclxuICAgIH1cclxuXHJcbiAgICAvLyBDcmVhdGUgaW5kZXggYnVmZmVyXHJcbiAgICBpZiAodHBsLmlkeCA9PSBudWxsKSB7XHJcbiAgICAgIHByaW0uaW5kZXhCdWZmZXIgPSBudWxsO1xyXG4gICAgICBwcmltLmluZGV4TnVtYmVyID0gMDtcclxuICAgIH0gZWxzZSB7XHJcbiAgICAgIHByaW0uaW5kZXhCdWZmZXIgPSBnbC5jcmVhdGVCdWZmZXIoKTtcclxuICAgICAgZ2wuYmluZEJ1ZmZlcihnbC5FTEVNRU5UX0FSUkFZX0JVRkZFUiwgcHJpbS5pbmRleEJ1ZmZlcik7XHJcbiAgICAgIGdsLmJ1ZmZlckRhdGEoZ2wuRUxFTUVOVF9BUlJBWV9CVUZGRVIsIHRwbC5nZXRJbmRpY2VzQXNVaW50QXJyYXkoKSwgZ2wuU1RBVElDX0RSQVcpO1xyXG4gICAgICBwcmltLmluZGV4TnVtYmVyID0gdHBsLmlkeC5sZW5ndGg7XHJcbiAgICB9XHJcblxyXG4gICAgcmV0dXJuIHByaW07XHJcbiAgfSAvKiBmcm9tQXJyYXkgKi9cclxufTsgLyogUHJpbWl0aXZlICovXHJcbiIsImltcG9ydCAqIGFzIG10aCBmcm9tIFwiLi9tdGguanNcIjtcclxuaW1wb3J0IHtUZXh0dXJlfSBmcm9tIFwiLi90ZXh0dXJlLmpzXCI7XHJcblxyXG5mdW5jdGlvbiBkZWNvZGVGcmFtZWJ1ZmZlclN0YXR1cyhzdGF0dXMpIHtcclxuICBzd2l0Y2ggKHN0YXR1cykge1xyXG4gICAgY2FzZSBXZWJHTDJSZW5kZXJpbmdDb250ZXh0LkZSQU1FQlVGRkVSX0NPTVBMRVRFOiAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gXCJjb21wbGV0ZVwiO1xyXG4gICAgY2FzZSBXZWJHTDJSZW5kZXJpbmdDb250ZXh0LkZSQU1FQlVGRkVSX0lOQ09NUExFVEVfQVRUQUNITUVOVDogICAgICAgICByZXR1cm4gXCJpbmNvbXBsZXRlIGF0dGFjaG1lbnRcIjtcclxuICAgIGNhc2UgV2ViR0wyUmVuZGVyaW5nQ29udGV4dC5GUkFNRUJVRkZFUl9JTkNPTVBMRVRFX0RJTUVOU0lPTlM6ICAgICAgICAgcmV0dXJuIFwiaGVpZ2h0IGFuZCB3aWR0aCBvZiBhdHRhY2htZW50IGFyZW4ndCBzYW1lXCI7XHJcbiAgICBjYXNlIFdlYkdMMlJlbmRlcmluZ0NvbnRleHQuRlJBTUVCVUZGRVJfSU5DT01QTEVURV9NSVNTSU5HX0FUVEFDSE1FTlQ6IHJldHVybiBcImF0dGFjaG1lbnQgbWlzc2luZ1wiO1xyXG4gICAgY2FzZSBXZWJHTDJSZW5kZXJpbmdDb250ZXh0LkZSQU1FQlVGRkVSX1VOU1VQUE9SVEVEOiAgICAgICAgICAgICAgICAgICByZXR1cm4gXCJhdHRhY2htZW50IGZvcm1hdCBpc24ndCBzdXBwb3J0ZWRcIjtcclxuICB9XHJcbn1cclxuXHJcbmV4cG9ydCBjbGFzcyBUYXJnZXQge1xyXG4gICNnbDtcclxuICBGQk87XHJcbiAgYXR0YWNobWVudHMgPSBbXTtcclxuICBzaXplO1xyXG4gIGRlcHRoO1xyXG5cclxuICBjb25zdHJ1Y3RvcihnbCwgYXR0YWNobWVudENvdW50KSB7XHJcbiAgICB0aGlzLnNpemUgPSBuZXcgbXRoLlNpemUoODAwLCA2MDApO1xyXG4gICAgdGhpcy5nbCA9IGdsO1xyXG4gICAgdGhpcy5GQk8gPSBnbC5jcmVhdGVGcmFtZWJ1ZmZlcigpO1xyXG5cclxuICAgIGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgdGhpcy5GQk8pO1xyXG5cclxuICAgIC8vIGNyZWF0ZSB0YXJnZXQgdGV4dHVyZXNcclxuICAgIGxldCBkcmF3QnVmZmVycyA9IFtdO1xyXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBhdHRhY2htZW50Q291bnQ7IGkrKykge1xyXG4gICAgICB0aGlzLmF0dGFjaG1lbnRzW2ldID0gbmV3IFRleHR1cmUoZ2wsIFRleHR1cmUuRkxPQVQsIDQpO1xyXG4gICAgICBkcmF3QnVmZmVycy5wdXNoKGdsLkNPTE9SX0FUVEFDSE1FTlQwICsgaSk7XHJcbiAgICB9XHJcbiAgICBnbC5kcmF3QnVmZmVycyhkcmF3QnVmZmVycyk7XHJcblxyXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBhdHRhY2htZW50Q291bnQ7IGkrKykge1xyXG4gICAgICBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0aGlzLmF0dGFjaG1lbnRzW2ldLmlkKTtcclxuICAgICAgdGhpcy5hdHRhY2htZW50c1tpXS5yZXNpemUodGhpcy5zaXplKTtcclxuICBcclxuICAgICAgZ2wuZnJhbWVidWZmZXJUZXh0dXJlMkQoZ2wuRlJBTUVCVUZGRVIsIGdsLkNPTE9SX0FUVEFDSE1FTlQwICsgaSwgZ2wuVEVYVFVSRV8yRCwgdGhpcy5hdHRhY2htZW50c1tpXS5pZCwgMCk7XHJcbiAgICB9XHJcbiAgICB0aGlzLmRlcHRoID0gbmV3IFRleHR1cmUoZ2wsIFRleHR1cmUuREVQVEgpO1xyXG4gICAgdGhpcy5kZXB0aC5yZXNpemUodGhpcy5zaXplKTtcclxuICAgIGdsLmJpbmRUZXh0dXJlKGdsLlRFWFRVUkVfMkQsIHRoaXMuZGVwdGguaWQpO1xyXG4gICAgZ2wuZnJhbWVidWZmZXJUZXh0dXJlMkQoZ2wuRlJBTUVCVUZGRVIsIGdsLkRFUFRIX0FUVEFDSE1FTlQsIGdsLlRFWFRVUkVfMkQsIHRoaXMuZGVwdGguaWQsIDApO1xyXG5cclxuICAgIGNvbnNvbGUubG9nKGBGcmFtZWJ1ZmZlciBzdGF0dXM6ICR7ZGVjb2RlRnJhbWVidWZmZXJTdGF0dXMoZ2wuY2hlY2tGcmFtZWJ1ZmZlclN0YXR1cyhnbC5GUkFNRUJVRkZFUikpfWApO1xyXG4gIH0gLyogY29uc3RydWN0b3IgKi9cclxuXHJcbiAgcmVzaXplKHNpemUpIHtcclxuICAgIGxldCBnbCA9IHRoaXMuZ2w7XHJcblxyXG4gICAgdGhpcy5zaXplID0gc2l6ZS5jb3B5KCk7XHJcbiAgICB0aGlzLkZCTyA9IGdsLmNyZWF0ZUZyYW1lYnVmZmVyKCk7XHJcblxyXG4gICAgZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCB0aGlzLkZCTyk7XHJcblxyXG4gICAgLy8gY3JlYXRlIHRhcmdldCB0ZXh0dXJlc1xyXG4gICAgbGV0IGRyYXdCdWZmZXJzID0gW107XHJcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMuYXR0YWNobWVudHMubGVuZ3RoOyBpKyspIHtcclxuICAgICAgdGhpcy5hdHRhY2htZW50c1tpXSA9IG5ldyBUZXh0dXJlKGdsLCBUZXh0dXJlLkZMT0FULCA0KTtcclxuICAgICAgZHJhd0J1ZmZlcnMucHVzaChnbC5DT0xPUl9BVFRBQ0hNRU5UMCArIGkpO1xyXG4gICAgfVxyXG4gICAgZ2wuZHJhd0J1ZmZlcnMoZHJhd0J1ZmZlcnMpO1xyXG5cclxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5hdHRhY2htZW50cy5sZW5ndGg7IGkrKykge1xyXG4gICAgICBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0aGlzLmF0dGFjaG1lbnRzW2ldLmlkKTtcclxuICAgICAgdGhpcy5hdHRhY2htZW50c1tpXS5yZXNpemUodGhpcy5zaXplKTtcclxuICBcclxuICAgICAgZ2wuZnJhbWVidWZmZXJUZXh0dXJlMkQoZ2wuRlJBTUVCVUZGRVIsIGdsLkNPTE9SX0FUVEFDSE1FTlQwICsgaSwgZ2wuVEVYVFVSRV8yRCwgdGhpcy5hdHRhY2htZW50c1tpXS5pZCwgMCk7XHJcbiAgICB9XHJcbiAgICB0aGlzLmRlcHRoID0gbmV3IFRleHR1cmUoZ2wsIFRleHR1cmUuREVQVEgpO1xyXG4gICAgdGhpcy5kZXB0aC5yZXNpemUodGhpcy5zaXplKTtcclxuICAgIGdsLmJpbmRUZXh0dXJlKGdsLlRFWFRVUkVfMkQsIHRoaXMuZGVwdGguaWQpO1xyXG4gICAgZ2wuZnJhbWVidWZmZXJUZXh0dXJlMkQoZ2wuRlJBTUVCVUZGRVIsIGdsLkRFUFRIX0FUVEFDSE1FTlQsIGdsLlRFWFRVUkVfMkQsIHRoaXMuZGVwdGguaWQsIDApO1xyXG5cclxuICAgIGNvbnNvbGUubG9nKGBGcmFtZWJ1ZmZlciBzdGF0dXM6ICR7ZGVjb2RlRnJhbWVidWZmZXJTdGF0dXMoZ2wuY2hlY2tGcmFtZWJ1ZmZlclN0YXR1cyhnbC5GUkFNRUJVRkZFUikpfWApO1xyXG4gIH0gLyogcmVzaXplICovXHJcblxyXG4gIGJpbmQoKSB7XHJcbiAgICBsZXQgZ2wgPSB0aGlzLmdsO1xyXG5cclxuICAgIGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgdGhpcy5GQk8pO1xyXG5cclxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5hdHRhY2htZW50cy5sZW5ndGg7IGkrKylcclxuICAgIGdsLmNsZWFyQnVmZmVyZnYoZ2wuQ09MT1IsIGksIFswLjAwLCAwLjAwLCAwLjAwLCAwLjAwXSk7XHJcbiAgICBnbC5jbGVhckJ1ZmZlcmZ2KGdsLkRFUFRILCAwLCBbMV0pO1xyXG4gICAgZ2wudmlld3BvcnQoMCwgMCwgdGhpcy5zaXplLncsIHRoaXMuc2l6ZS5oKTtcclxuICB9IC8qIGJpbmQgKi9cclxuXHJcbiAgc3RhdGljIGRlZmF1bHRGcmFtZWJ1ZmZlciA9IHtcclxuICAgIHNpemU6IG5ldyBtdGguU2l6ZSg4MDAsIDYwMCksXHJcbiAgICBnbDogbnVsbCxcclxuXHJcbiAgICByZXNpemUoc2l6ZSkge1xyXG4gICAgICBUYXJnZXQuZGVmYXVsdEZyYW1lYnVmZmVyLnNpemUgPSBzaXplLmNvcHkoKTtcclxuICAgIH0sIC8qIHJlc2l6ZSAqL1xyXG5cclxuICAgIGJpbmQoKSB7XHJcbiAgICAgIGxldCBnbCA9IFRhcmdldC5kZWZhdWx0RnJhbWVidWZmZXIuZ2w7XHJcblxyXG4gICAgICBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIG51bGwpO1xyXG4gICAgICBnbC52aWV3cG9ydCgwLCAwLCBUYXJnZXQuZGVmYXVsdEZyYW1lYnVmZmVyLnNpemUudywgVGFyZ2V0LmRlZmF1bHRGcmFtZWJ1ZmZlci5zaXplLmgpO1xyXG4gICAgICBnbC5jbGVhcihnbC5DT0xPUl9CVUZGRVJfQklUIHwgZ2wuREVQVEhfQlVGRkVSX0JJVCk7XHJcbiAgICAgIGdsLmNsZWFyQ29sb3IoMC4zMCwgMC40NywgMC44MCwgMS4wMCk7XHJcbiAgICB9XHJcbiAgfTsgLyogZGVmYXVsdEZyYW1lYnVmZmVyICovXHJcblxyXG4gIHN0YXRpYyBkZWZhdWx0KGdsKSB7XHJcbiAgICBUYXJnZXQuZGVmYXVsdEZyYW1lYnVmZmVyLmdsID0gZ2w7XHJcbiAgICByZXR1cm4gVGFyZ2V0LmRlZmF1bHRGcmFtZWJ1ZmZlcjtcclxuICB9IC8qIGRlZmF1bHQgKi9cclxufSAvKiBUYXJnZXQgKi9cclxuXHJcbi8qIHRhcmdldC5qcyAqLyIsImltcG9ydCB7bG9hZFNoYWRlciwgTWF0ZXJpYWwsIFByaW1pdGl2ZSwgVG9wb2xvZ3ksIFZlcnRleCwgVGV4dHVyZSwgQ3ViZW1hcCwgVUJPLCBtdGh9IGZyb20gXCIuL3ByaW1pdGl2ZS5qc1wiO1xyXG5pbXBvcnQge1RhcmdldH0gZnJvbSBcIi4vdGFyZ2V0LmpzXCI7XHJcblxyXG5leHBvcnQge01hdGVyaWFsLCBQcmltaXRpdmUsIFRvcG9sb2d5LCBWZXJ0ZXgsIFRleHR1cmUsIEN1YmVtYXAsIFVCTywgbXRofTtcclxuXHJcbmV4cG9ydCBjbGFzcyBSZW5kZXIge1xyXG4gIHJlbmRlclF1ZXVlO1xyXG4gIGdsO1xyXG4gIGNhbWVyYTtcclxuICBjYW1lcmFVQk87XHJcblxyXG4gIHRhcmdldDtcclxuICBmc1ByaW1pdGl2ZSA9IG51bGw7XHJcblxyXG4gIGNvbnN0cnVjdG9yKCkge1xyXG4gICAgLy8gV2ViR0wgaW5pdGlhbGl6YXRpb25cclxuICAgIGxldCBjYW52YXMgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcImdsQ2FudmFzXCIpO1xyXG5cclxuICAgIGNhbnZhcy53aWR0aCA9IHdpbmRvdy5pbm5lcldpZHRoO1xyXG4gICAgY2FudmFzLmhlaWdodCA9IHdpbmRvdy5pbm5lckhlaWdodDtcclxuICAgIGxldCBnbCA9IGNhbnZhcy5nZXRDb250ZXh0KFwid2ViZ2wyXCIpO1xyXG4gICAgaWYgKGdsID09IG51bGwpIHtcclxuICAgICAgdGhyb3cgRXJyb3IoXCJDYW4ndCBpbml0aWFsaXplIFdlYkdMMlwiKTtcclxuICAgIH1cclxuXHJcbiAgICBsZXQgZXh0ZW5zaW9ucyA9IFtcIkVYVF9jb2xvcl9idWZmZXJfZmxvYXRcIiwgXCJPRVNfdGV4dHVyZV9mbG9hdF9saW5lYXJcIl07XHJcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGV4dGVuc2lvbnMubGVuZ3RoOyBpKyspXHJcbiAgICAgIGlmIChnbC5nZXRFeHRlbnNpb24oZXh0ZW5zaW9uc1tpXSkgPT0gbnVsbClcclxuICAgICAgICB0aHJvdyBFcnJvcihgXCIke2V4dGVuc2lvbnNbaV19XCIgZXh0ZW5zaW9uIHJlcXVpcmVkYCk7XHJcblxyXG4gICAgdGhpcy5nbCA9IGdsO1xyXG5cclxuICAgIGdsLmVuYWJsZShXZWJHTDJSZW5kZXJpbmdDb250ZXh0LkRFUFRIX1RFU1QpO1xyXG5cclxuICAgIHRoaXMucmVuZGVyUXVldWUgPSBbXTtcclxuICAgIHRoaXMuY2FtZXJhID0gbmV3IG10aC5DYW1lcmEoKTtcclxuXHJcbiAgICB0aGlzLmNhbWVyYVVCTyA9IG5ldyBVQk8odGhpcy5nbCk7XHJcblxyXG4gICAgdGhpcy5jYW1lcmEucmVzaXplKG5ldyBtdGguU2l6ZShjYW52YXMud2lkdGgsIGNhbnZhcy5oZWlnaHQpKTtcclxuXHJcbiAgICAvLyB0YXJnZXRzIHNldHVwXHJcbiAgICBsZXQgc2l6ZSA9IG5ldyBtdGguU2l6ZShjYW52YXMud2lkdGgsIGNhbnZhcy5oZWlnaHQpO1xyXG4gICAgdGhpcy50YXJnZXQgPSBuZXcgVGFyZ2V0KGdsLCAzKTtcclxuXHJcbiAgICB0aGlzLnRhcmdldC5yZXNpemUoc2l6ZSk7XHJcbiAgICBUYXJnZXQuZGVmYXVsdChnbCkucmVzaXplKHNpemUpO1xyXG5cclxuICAgIC8vIHJlc2l6ZSBoYW5kbGluZ1xyXG4gICAgd2luZG93Lm9ucmVzaXplID0gKCkgPT4ge1xyXG4gICAgICBsZXQgcmVzb2x1dGlvbiA9IG5ldyBtdGguU2l6ZSh3aW5kb3cuaW5uZXJXaWR0aCwgd2luZG93LmlubmVySGVpZ2h0KTtcclxuXHJcbiAgICAgIGNhbnZhcy53aWR0aCA9IHJlc29sdXRpb24udztcclxuICAgICAgY2FudmFzLmhlaWdodCA9IHJlc29sdXRpb24uaDtcclxuXHJcbiAgICAgIHRoaXMuY2FtZXJhLnJlc2l6ZShyZXNvbHV0aW9uKTtcclxuICAgICAgdGhpcy50YXJnZXQucmVzaXplKHJlc29sdXRpb24pO1xyXG4gICAgICBUYXJnZXQuZGVmYXVsdChnbCkucmVzaXplKHJlc29sdXRpb24pO1xyXG4gICAgfTtcclxuICB9IC8qIGNvbnN0cnVjdG9yICovXHJcblxyXG4gIGRyYXdQcmltaXRpdmUocHJpbWl0aXZlLCB0cmFuc2Zvcm0gPSBtdGguTWF0NC5pZGVudGl0eSgpKSB7XHJcbiAgICB0aGlzLnJlbmRlclF1ZXVlLnB1c2goe3ByaW1pdGl2ZSwgdHJhbnNmb3JtfSk7XHJcbiAgfSAvKiBkcmF3UHJpbWl0aXZlICovXHJcblxyXG4gIGNyZWF0ZVRleHR1cmUoKSB7XHJcbiAgICByZXR1cm4gbmV3IFRleHR1cmUodGhpcy5nbCwgVGV4dHVyZS5VTlNJR05FRF9CWVRFLCA0KTtcclxuICB9IC8qIGNyZWF0ZVRleHR1cmUgKi9cclxuXHJcbiAgY3JlYXRlQ3ViZW1hcCgpIHtcclxuICAgIHJldHVybiBuZXcgQ3ViZW1hcCh0aGlzLmdsKTtcclxuICB9IC8qIGNyZWF0ZUN1YmVtYXAgKi9cclxuXHJcbiAgY3JlYXRlVW5pZm9ybUJ1ZmZlcigpIHtcclxuICAgIHJldHVybiBuZXcgVUJPKHRoaXMuZ2wpO1xyXG4gIH0gLyogY3JlYXRlVW5pZm9ybUJ1ZmZlciAqL1xyXG5cclxuICBhc3luYyBjcmVhdGVTaGFkZXIocGF0aCkge1xyXG4gICAgcmV0dXJuIGxvYWRTaGFkZXIodGhpcy5nbCwgcGF0aCk7XHJcbiAgfSAvKiBjcmVhdGVTaGFkZXIgKi9cclxuXHJcbiAgYXN5bmMgY3JlYXRlTWF0ZXJpYWwoc2hhZGVyKSB7XHJcbiAgICBpZiAodHlwZW9mKHNoYWRlcikgPT0gXCJzdHJpbmdcIikge1xyXG4gICAgICByZXR1cm4gbmV3IE1hdGVyaWFsKHRoaXMuZ2wsIGF3YWl0IGxvYWRTaGFkZXIodGhpcy5nbCwgc2hhZGVyKSk7XHJcbiAgICB9IGVsc2Uge1xyXG4gICAgICByZXR1cm4gbmV3IE1hdGVyaWFsKHRoaXMuZ2wsIHNoYWRlcik7XHJcbiAgICB9XHJcbiAgfSAvKiBjcmVhdGVNYXRlcmlhbCAqL1xyXG5cclxuICBjcmVhdGVQcmltaXRpdmUodG9wb2xvZ3ksIG1hdGVyaWFsKSB7XHJcbiAgICByZXR1cm4gUHJpbWl0aXZlLmZyb21Ub3BvbG9neSh0aGlzLmdsLCB0b3BvbG9neSwgbWF0ZXJpYWwpO1xyXG4gIH0gLyogY3JlYXRlUHJpbWl0aXZlICovXHJcblxyXG4gIGFzeW5jIHN0YXJ0KCkge1xyXG4gICAgaWYgKHRoaXMuZnNQcmltaXRpdmUgPT0gbnVsbCkge1xyXG4gICAgICB0aGlzLmZzUHJpbWl0aXZlID0gYXdhaXQgdGhpcy5jcmVhdGVQcmltaXRpdmUoVG9wb2xvZ3kuc3F1YXJlKCksIGF3YWl0IHRoaXMuY3JlYXRlTWF0ZXJpYWwoXCIuL2Jpbi9zaGFkZXJzL3RhcmdldFwiKSk7XHJcbiAgICAgIHRoaXMuZnNQcmltaXRpdmUubWF0ZXJpYWwudGV4dHVyZXMgPSB0aGlzLnRhcmdldC5hdHRhY2htZW50cztcclxuICAgIH1cclxuICB9IC8qIHN0YXJ0ICovXHJcbiAgXHJcbiAgZW5kKCkge1xyXG4gICAgLy8gcmVuZGVyaW5nIGluIHRhcmdldFxyXG4gICAgbGV0IGdsID0gdGhpcy5nbDtcclxuXHJcbiAgICB0aGlzLnRhcmdldC5iaW5kKCk7XHJcblxyXG4gICAgbGV0IGNhbWVyYUluZm8gPSBuZXcgRmxvYXQzMkFycmF5KDM2KTtcclxuXHJcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IDE2OyBpKyspIHtcclxuICAgICAgY2FtZXJhSW5mb1tpICsgMTZdID0gdGhpcy5jYW1lcmEudmlld1Byb2oubVtpXTtcclxuICAgIH1cclxuICAgIGNhbWVyYUluZm9bMzJdID0gdGhpcy5jYW1lcmEubG9jLng7XHJcbiAgICBjYW1lcmFJbmZvWzMzXSA9IHRoaXMuY2FtZXJhLmxvYy55O1xyXG4gICAgY2FtZXJhSW5mb1szNF0gPSB0aGlzLmNhbWVyYS5sb2MuejtcclxuXHJcbiAgICBmb3IgKGxldCBpID0gMCwgY291bnQgPSB0aGlzLnJlbmRlclF1ZXVlLmxlbmd0aDsgaSA8IGNvdW50OyBpKyspIHtcclxuICAgICAgbGV0IHByaW0gPSB0aGlzLnJlbmRlclF1ZXVlW2ldLnByaW1pdGl2ZTtcclxuICAgICAgbGV0IHRyYW5zID0gdGhpcy5yZW5kZXJRdWV1ZVtpXS50cmFuc2Zvcm07XHJcblxyXG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IDE2OyBpKyspIHtcclxuICAgICAgICBjYW1lcmFJbmZvW2ldID0gdHJhbnMubVtpXTtcclxuICAgICAgfVxyXG5cclxuICAgICAgdGhpcy5jYW1lcmFVQk8ud3JpdGVEYXRhKGNhbWVyYUluZm8pO1xyXG5cclxuICAgICAgcHJpbS5kcmF3KHRoaXMuY2FtZXJhVUJPKTtcclxuICAgIH1cclxuXHJcbiAgICAvLyBmbHVzaCByZW5kZXIgcXVldWVcclxuICAgIHRoaXMucmVuZGVyUXVldWUgPSBbXTtcclxuXHJcbiAgICAvLyByZW5kZXJpbmcgdG8gc2NyZWVuIGZyYW1lYnVmZmVyXHJcbiAgICBUYXJnZXQuZGVmYXVsdChnbCkuYmluZCgpO1xyXG4gICAgdGhpcy5mc1ByaW1pdGl2ZS5kcmF3KHRoaXMuY2FtZXJhVUJPKTtcclxuICB9IC8qIGVuZCAqL1xyXG59IC8qIFJlbmRlciAqL1xyXG5cclxuLyogcmVuZGVyLmpzICovIiwiZnVuY3Rpb24gZ2V0VGltZSgpIHtcclxuICByZXR1cm4gRGF0ZS5ub3coKSAvIDEwMDAuMDtcclxufVxyXG5cclxuZXhwb3J0IGNsYXNzIFRpbWVyIHtcclxuICBmcHNQcmV2VXBkYXRlVGltZSA9IDAuMDA7XHJcbiAgc3RhcnRUaW1lO1xyXG4gIGZwc0NvdW50ZXIgPSAwLjAwO1xyXG4gIHBhdXNlQ29sbGVjdG9yID0gMC4wMDtcclxuICBpc1BhdXNlZCA9IGZhbHNlO1xyXG5cclxuICBmcHNEZWx0YVRpbWUgPSAzLjAwO1xyXG4gIGZwcyA9IHVuZGVmaW5lZDtcclxuXHJcbiAgdGltZSA9IDAuMDA7XHJcbiAgZ2xvYmFsVGltZTtcclxuICBcclxuICBkZWx0YVRpbWUgPSAwLjAwO1xyXG4gIGdsb2JhbERlbHRhVGltZSA9IDAuMDA7XHJcblxyXG4gIGNvbnN0cnVjdG9yKCkge1xyXG4gICAgdGhpcy5zdGFydFRpbWUgPSBnZXRUaW1lKCk7XHJcblxyXG4gICAgdGhpcy5nbG9iYWxUaW1lID0gdGhpcy5zdGFydFRpbWU7XHJcbiAgfSAvKiBjb25zdHJ1Y3RvciAqL1xyXG5cclxuICByZXNwb25zZSgpIHtcclxuICAgIGxldCBuZXdHbG9iYWxUaW1lID0gZ2V0VGltZSgpO1xyXG5cclxuICAgIHRoaXMuZ2xvYmFsRGVsdGFUaW1lID0gbmV3R2xvYmFsVGltZSAtIHRoaXMuZ2xvYmFsVGltZTtcclxuICAgIHRoaXMuZ2xvYmFsVGltZSA9IG5ld0dsb2JhbFRpbWU7XHJcblxyXG4gICAgaWYgKHRoaXMuaXNQYXVzZWQpIHtcclxuICAgICAgdGhpcy5kZWx0YVRpbWUgPSAwLjAwO1xyXG4gICAgICB0aGlzLnBhdXNlQ29sbGVjdG9yICs9IHRoaXMuZ2xvYmFsRGVsdGFUaW1lO1xyXG4gICAgfSBlbHNlIHtcclxuICAgICAgdGhpcy50aW1lID0gdGhpcy5nbG9iYWxUaW1lIC0gdGhpcy5zdGFydFRpbWUgLSB0aGlzLnBhdXNlQ29sbGVjdG9yO1xyXG4gICAgICB0aGlzLmRlbHRhVGltZSA9IHRoaXMuZ2xvYmFsRGVsdGFUaW1lO1xyXG4gICAgfVxyXG5cclxuICAgIHRoaXMuZnBzQ291bnRlcisrO1xyXG4gICAgaWYgKHRoaXMuZ2xvYmFsVGltZSAtIHRoaXMuZnBzUHJldlVwZGF0ZVRpbWUgPj0gdGhpcy5mcHNEZWx0YVRpbWUpIHtcclxuICAgICAgdGhpcy5mcHMgPSB0aGlzLmZwc0NvdW50ZXIgLyAodGhpcy5nbG9iYWxUaW1lIC0gdGhpcy5mcHNQcmV2VXBkYXRlVGltZSk7XHJcblxyXG4gICAgICB0aGlzLmZwc1ByZXZVcGRhdGVUaW1lID0gdGhpcy5nbG9iYWxUaW1lO1xyXG4gICAgICB0aGlzLmZwc0NvdW50ZXIgPSAwO1xyXG4gICAgfVxyXG4gIH0gLyogcmVzcG9uc2UgKi9cclxufSAvKiBUaW1lciAqL1xyXG5cclxuLyogdGltZXIuanMgKi8iLCJpbXBvcnQgKiBhcyBybmQgZnJvbSBcIi4vcmVuZGVyLmpzXCI7XHJcbmltcG9ydCB7VGltZXJ9IGZyb20gXCIuL3RpbWVyLmpzXCI7XHJcblxyXG5leHBvcnQge3JuZH07XHJcblxyXG5leHBvcnQgY2xhc3MgU3lzdGVtIHtcclxuICByZW5kZXI7IC8vIFJlbmRlclxyXG4gIHVuaXRzOyAgLy8gdW5pdCBsaXN0XHJcbiAgdGltZXI7ICAvLyB0aW1lclxyXG5cclxuICBjb25zdHJ1Y3RvcigpIHtcclxuICAgIHRoaXMucmVuZGVyID0gbmV3IHJuZC5SZW5kZXIoKTtcclxuICAgIHRoaXMudGltZXIgPSBuZXcgVGltZXIoKTtcclxuICAgIHRoaXMudW5pdHMgPSBbXTtcclxuICB9IC8qIGNvbnN0cnVjdG9yICovXHJcblxyXG4gIHN0YXRpYyBpc1Byb21pc2UgPSBmdW5jdGlvbih2KSB7XHJcbiAgICByZXR1cm4gdiA9PiB0eXBlb2YodikgPT09IFwib2JqZWN0XCIgJiYgdHlwZW9mIHYudGhlbiA9PT0gXCJmdW5jdGlvblwiO1xyXG4gIH0gLyogaXNQcm9taXNlICovXHJcblxyXG4gIGFzeW5jIGFkZFVuaXQoY3JlYXRlRnVuY3Rpb24pIHtcclxuICAgIGxldCB2YWwgPSBjcmVhdGVGdW5jdGlvbih0aGlzKTtcclxuXHJcbiAgICBpZiAoU3lzdGVtLmlzUHJvbWlzZSh2YWwpKSB7XHJcbiAgICAgIHZhbCA9IGF3YWl0IHZhbDtcclxuICAgIH1cclxuXHJcbiAgICB0aGlzLnVuaXRzLnB1c2godmFsKTtcclxuXHJcbiAgICByZXR1cm4gdmFsO1xyXG4gIH0gLyogYWRkVW5pdCAqL1xyXG5cclxuICBydW4oKSB7XHJcbiAgICBsZXQgc3lzdGVtID0gdGhpcztcclxuXHJcbiAgICBjb25zdCBydW4gPSBhc3luYyBmdW5jdGlvbigpIHtcclxuICAgICAgc3lzdGVtLnRpbWVyLnJlc3BvbnNlKCk7XHJcblxyXG4gICAgICBhd2FpdCBzeXN0ZW0ucmVuZGVyLnN0YXJ0KCk7XHJcblxyXG4gICAgICBmb3IgKGxldCBpID0gMCwgY291bnQgPSBzeXN0ZW0udW5pdHMubGVuZ3RoOyBpIDwgY291bnQ7IGkrKykge1xyXG4gICAgICAgIHN5c3RlbS51bml0c1tpXS5yZXNwb25zZShzeXN0ZW0pO1xyXG4gICAgICB9XHJcblxyXG4gICAgICBzeXN0ZW0ucmVuZGVyLmVuZCgpO1xyXG5cclxuICAgICAgd2luZG93LnJlcXVlc3RBbmltYXRpb25GcmFtZShydW4pO1xyXG4gICAgfTtcclxuICAgIHdpbmRvdy5yZXF1ZXN0QW5pbWF0aW9uRnJhbWUocnVuKTtcclxuICB9IC8qIHJ1biAqL1xyXG59IC8qIFN5c3RlbSAqLyIsImltcG9ydCAqIGFzIHRjZ2wgZnJvbSBcIi4vLi4vc3lzdGVtLmpzXCI7XHJcbmltcG9ydCAqIGFzIG10aCBmcm9tIFwiLi8uLi9tdGguanNcIjtcclxuXHJcbi8vIGVsZXZhdGlvbiBvZiBpY29zYWhlZHJvbiBiYXNpYyBzcGhlcmVcclxuY29uc3QgaWNvc2FoZWRyb25FbGV2YXRpb24gPSAxLjEwNzE0ODcxNzc5NDA5MDQ7XHJcblxyXG4vLyBpY29zYWhlZHJvbiBldmFsdWF0aW9uXHJcbmxldCBpY28gPSBuZXcgdGNnbC5ybmQuVG9wb2xvZ3koKTtcclxuaWNvLnR5cGUgPSB0Y2dsLnJuZC5Ub3BvbG9neS5MSU5FUztcclxuXHJcbnRjZ2wucm5kLlZlcnRleC5mcm9tQ29vcmQoMCwgMSwgMCk7XHJcblxyXG5pY28udnR4ID0gW107XHJcbmljby52dHgucHVzaCh0Y2dsLnJuZC5WZXJ0ZXguZnJvbUNvb3JkKDAsIDEsIDApKTtcclxuZm9yIChsZXQgYSA9IDA7IGEgPCA1OyBhKyspIHtcclxuICBpY28udnR4LnB1c2godGNnbC5ybmQuVmVydGV4LmZyb21WZWN0b3JzKG10aC5WZWMzLmZyb21TcGhlcmljYWwoYSAqIE1hdGguUEkgKiAwLjQsIGljb3NhaGVkcm9uRWxldmF0aW9uLCAxKSkpO1xyXG59XHJcbmZvciAobGV0IGEgPSAwOyBhIDwgNTsgYSsrKSB7XHJcbiAgaWNvLnZ0eC5wdXNoKHRjZ2wucm5kLlZlcnRleC5mcm9tVmVjdG9ycyhtdGguVmVjMy5mcm9tU3BoZXJpY2FsKGEgKiBNYXRoLlBJICogMC40ICsgTWF0aC5QSSAvIDUuMCwgTWF0aC5QSSAtIGljb3NhaGVkcm9uRWxldmF0aW9uLCAxKSkpO1xyXG59XHJcbmljby52dHgucHVzaCh0Y2dsLnJuZC5WZXJ0ZXguZnJvbUNvb3JkKDAsIC0xLCAwKSk7XHJcblxyXG4vLyB3cml0ZSB3aXJlIGluZGljZXNcclxuaWNvLmlkeCA9IFtdO1xyXG5mb3IgKGxldCBhID0gMDsgYSA8IDU7IGErKykge1xyXG4gIGljby5pZHgucHVzaCgwKTtcclxuICBpY28uaWR4LnB1c2goYSArIDEpO1xyXG5cclxuICBpY28uaWR4LnB1c2goYSArIDEpO1xyXG4gIGljby5pZHgucHVzaCgoYSArIDEpICUgNSArIDEpO1xyXG5cclxuICBpY28uaWR4LnB1c2goKGEgKyAxKSAlIDUgKyAxKTtcclxuICBpY28uaWR4LnB1c2goYSArIDYpO1xyXG5cclxuICBpY28uaWR4LnB1c2goYSArIDEpO1xyXG4gIGljby5pZHgucHVzaChhICsgNik7XHJcblxyXG4gIGljby5pZHgucHVzaChhICsgNik7XHJcbiAgaWNvLmlkeC5wdXNoKChhICsgMSkgJSA1ICsgNik7XHJcblxyXG4gIGljby5pZHgucHVzaCgxMSk7XHJcbiAgaWNvLmlkeC5wdXNoKGEgKyA2KTtcclxufVxyXG5cclxuLy8gd3JpdGUgcG9seWdvbmFsIGluZGljZXNcclxuaWNvLnBvbHlnb25faWR4ID0gW107XHJcbmZvciAobGV0IHkgPSAwOyB5IDwgNTsgeSsrKSB7XHJcbiAgaWNvLnBvbHlnb25faWR4LnB1c2goMCk7XHJcbiAgaWNvLnBvbHlnb25faWR4LnB1c2goeSArIDEpO1xyXG4gIGljby5wb2x5Z29uX2lkeC5wdXNoKCh5ICsgMSkgJSA1ICsgMSk7XHJcblxyXG4gIGljby5wb2x5Z29uX2lkeC5wdXNoKHkgKyAxKTtcclxuICBpY28ucG9seWdvbl9pZHgucHVzaCgoeSArIDEpICUgNSArIDEpO1xyXG4gIGljby5wb2x5Z29uX2lkeC5wdXNoKHkgKyA2KTtcclxuXHJcbiAgaWNvLnBvbHlnb25faWR4LnB1c2goeSArIDYpO1xyXG4gIGljby5wb2x5Z29uX2lkeC5wdXNoKCh5ICsgMSkgJSA1ICsgMSk7XHJcbiAgaWNvLnBvbHlnb25faWR4LnB1c2goKHkgKyAxKSAlIDUgKyA2KTtcclxuXHJcbiAgaWNvLnBvbHlnb25faWR4LnB1c2goKHkgKyAxKSAlIDUgKyA2KTtcclxuICBpY28ucG9seWdvbl9pZHgucHVzaCh5ICsgNik7XHJcbiAgaWNvLnBvbHlnb25faWR4LnB1c2goMTEpO1xyXG59XHJcblxyXG5cclxuLy8gZG9kZWNhaGVkcm9uIGV2YWx1YXRpb25cclxubGV0IGRvZCA9IG5ldyB0Y2dsLnJuZC5Ub3BvbG9neShbXSwgW10pO1xyXG5kb2QudHlwZSA9IHRjZ2wucm5kLlRvcG9sb2d5LkxJTkVTO1xyXG5cclxuZm9yIChsZXQgaSA9IDA7IGkgPCBpY28ucG9seWdvbl9pZHgubGVuZ3RoIC8gMzsgaSsrKSB7XHJcbiAgbGV0IHAxID0gaWNvLnZ0eFtpY28ucG9seWdvbl9pZHhbaSAqIDMgKyAwXV0ucDtcclxuICBsZXQgcDIgPSBpY28udnR4W2ljby5wb2x5Z29uX2lkeFtpICogMyArIDFdXS5wO1xyXG4gIGxldCBwMyA9IGljby52dHhbaWNvLnBvbHlnb25faWR4W2kgKiAzICsgMl1dLnA7XHJcblxyXG4gIGRvZC52dHgucHVzaCh0Y2dsLnJuZC5WZXJ0ZXguZnJvbVZlY3RvcnMocDEuYWRkKHAyKS5hZGQocDMpLm11bCgxLjAgLyAzKSkpO1xyXG59XHJcblxyXG5mb3IgKGxldCBhID0gMDsgYSA8IDU7IGErKykge1xyXG4gIGRvZC5pZHgucHVzaChhICogNCArIDApO1xyXG4gIGRvZC5pZHgucHVzaChhICogNCArIDEpO1xyXG5cclxuICBkb2QuaWR4LnB1c2goYSAqIDQgKyAxKTtcclxuICBkb2QuaWR4LnB1c2goYSAqIDQgKyAyKTtcclxuXHJcbiAgZG9kLmlkeC5wdXNoKGEgKiA0ICsgMik7XHJcbiAgZG9kLmlkeC5wdXNoKGEgKiA0ICsgMyk7XHJcblxyXG4gIGRvZC5pZHgucHVzaChhICogNCArIDApO1xyXG4gIGRvZC5pZHgucHVzaCgoYSArIDEpICUgNSAqIDQgKyAwKTtcclxuXHJcbiAgZG9kLmlkeC5wdXNoKGEgKiA0ICsgMik7XHJcbiAgZG9kLmlkeC5wdXNoKChhICsgMSkgJSA1ICogNCArIDEpO1xyXG5cclxuICBkb2QuaWR4LnB1c2goYSAqIDQgKyAzKTtcclxuICBkb2QuaWR4LnB1c2goKGEgKyAxKSAlIDUgKiA0ICsgMyk7XHJcbn1cclxuXHJcbi8vIGN1YmUgZXZhbHVhdGlvblxyXG5sZXQgY3ViID0gbmV3IHRjZ2wucm5kLlRvcG9sb2d5KCk7XHJcbmN1Yi50eXBlID0gdGNnbC5ybmQuVG9wb2xvZ3kuTElORVM7XHJcbmN1Yi52dHggPSBbXHJcbiAgdGNnbC5ybmQuVmVydGV4LmZyb21Db29yZCgtMSwgLTEsIC0xKSxcclxuICB0Y2dsLnJuZC5WZXJ0ZXguZnJvbUNvb3JkKC0xLCAgMSwgLTEpLFxyXG4gIHRjZ2wucm5kLlZlcnRleC5mcm9tQ29vcmQoIDEsICAxLCAtMSksXHJcbiAgdGNnbC5ybmQuVmVydGV4LmZyb21Db29yZCggMSwgLTEsIC0xKSxcclxuXHJcbiAgdGNnbC5ybmQuVmVydGV4LmZyb21Db29yZCgtMSwgLTEsICAxKSxcclxuICB0Y2dsLnJuZC5WZXJ0ZXguZnJvbUNvb3JkKC0xLCAgMSwgIDEpLFxyXG4gIHRjZ2wucm5kLlZlcnRleC5mcm9tQ29vcmQoIDEsICAxLCAgMSksXHJcbiAgdGNnbC5ybmQuVmVydGV4LmZyb21Db29yZCggMSwgLTEsICAxKSxcclxuXTtcclxuXHJcbmN1Yi5pZHggPSBbXHJcbiAgMCwgMSwgMSwgMiwgMiwgMywgMywgMCxcclxuICA0LCA1LCA1LCA2LCA2LCA3LCA3LCA0LFxyXG4gIDAsIDQsIDEsIDUsIDIsIDYsIDMsIDdcclxuXTtcclxuXHJcbi8vIG9jdGFoZWRyb24gZXZhbHVhdGlvblxyXG5sZXQgb2N0ID0gbmV3IHRjZ2wucm5kLlRvcG9sb2d5KCk7XHJcbm9jdC50eXBlID0gdGNnbC5ybmQuVG9wb2xvZ3kuTElORVM7XHJcblxyXG5vY3QudnR4ID0gW1xyXG4gIHRjZ2wucm5kLlZlcnRleC5mcm9tQ29vcmQoIDAsICAxLCAgMCksXHJcblxyXG4gIHRjZ2wucm5kLlZlcnRleC5mcm9tQ29vcmQoIDEsICAwLCAgMCksXHJcbiAgdGNnbC5ybmQuVmVydGV4LmZyb21Db29yZCggMCwgIDAsICAxKSxcclxuICB0Y2dsLnJuZC5WZXJ0ZXguZnJvbUNvb3JkKC0xLCAgMCwgIDApLFxyXG4gIHRjZ2wucm5kLlZlcnRleC5mcm9tQ29vcmQoIDAsICAwLCAtMSksXHJcblxyXG4gIHRjZ2wucm5kLlZlcnRleC5mcm9tQ29vcmQoIDAsIC0xLCAgMCksXHJcbl07XHJcbm9jdC5pZHggPSBbXHJcbiAgMCwgMSwgMCwgMiwgMCwgMywgMCwgNCxcclxuICAxLCAyLCAyLCAzLCAzLCA0LCA0LCAxLFxyXG4gIDUsIDEsIDUsIDIsIDUsIDMsIDUsIDRcclxuXTtcclxuXHJcbi8vIHRldHJhaGVkcm9uIGV2YWx1dGF0aW9uXHJcbmNvbnN0IHRldEhlaWdodCA9IE1hdGguc3FydCgyIC8gMyk7XHJcbmNvbnN0IHRldElubmVyTGVuID0gMSAvIE1hdGguc3FydCgzKTtcclxuXHJcbmxldCB0ZXQgPSBuZXcgdGNnbC5ybmQuVG9wb2xvZ3koKTtcclxudGV0LnR5cGUgPSB0Y2dsLnJuZC5Ub3BvbG9neS5MSU5FUztcclxudGV0LnZ0eCA9IFt0Y2dsLnJuZC5WZXJ0ZXguZnJvbVZlY3RvcnMobmV3IG10aC5WZWMzKDAsIHRldEhlaWdodCwgMCkpXTtcclxuZm9yIChsZXQgYSA9IDA7IGEgPCAzOyBhKyspIHtcclxuICB0ZXQudnR4LnB1c2godGNnbC5ybmQuVmVydGV4LmZyb21WZWN0b3JzKG5ldyBtdGguVmVjMyhcclxuICAgIE1hdGguY29zKDIgKiBNYXRoLlBJIC8gMyAqIGEpICogdGV0SW5uZXJMZW4gKiAyLFxyXG4gICAgLXRldEhlaWdodCxcclxuICAgIE1hdGguc2luKDIgKiBNYXRoLlBJIC8gMyAqIGEpICogdGV0SW5uZXJMZW4gKiAyXHJcbiAgKSkpO1xyXG59XHJcbnRldC5pZHggPSBbMCwgMSwgMCwgMiwgMCwgMywgMSwgMiwgMiwgMywgMywgMV07XHJcblxyXG4vLyBhZGQgcGxhdG9uaWMgYm9kaWVzIHVuaXRzXHJcbmV4cG9ydCBhc3luYyBmdW5jdGlvbiBjcmVhdGUoc3lzdGVtKSB7XHJcbiAgbGV0IHBsYXRvbmljTXRsID0gYXdhaXQgc3lzdGVtLnJlbmRlci5jcmVhdGVNYXRlcmlhbChcIi4vYmluL3NoYWRlcnMvcGxhdG9uaWNcIik7XHJcblxyXG4gIGxldCBwcmltcyA9IFtcclxuICAgIGF3YWl0IHN5c3RlbS5yZW5kZXIuY3JlYXRlUHJpbWl0aXZlKGljbywgcGxhdG9uaWNNdGwpLFxyXG4gICAgYXdhaXQgc3lzdGVtLnJlbmRlci5jcmVhdGVQcmltaXRpdmUoZG9kLCBwbGF0b25pY010bCksXHJcbiAgICBhd2FpdCBzeXN0ZW0ucmVuZGVyLmNyZWF0ZVByaW1pdGl2ZShjdWIsIHBsYXRvbmljTXRsKSxcclxuICAgIGF3YWl0IHN5c3RlbS5yZW5kZXIuY3JlYXRlUHJpbWl0aXZlKG9jdCwgcGxhdG9uaWNNdGwpLFxyXG4gICAgYXdhaXQgc3lzdGVtLnJlbmRlci5jcmVhdGVQcmltaXRpdmUodGV0LCBwbGF0b25pY010bClcclxuICBdO1xyXG5cclxuICByZXR1cm4ge1xyXG4gICAgcmVzcG9uc2Uoc3lzdGVtKSB7XHJcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgcHJpbXMubGVuZ3RoOyBpKyspIHtcclxuICAgICAgICBzeXN0ZW0ucmVuZGVyLmRyYXdQcmltaXRpdmUocHJpbXNbaV0sXHJcbiAgICAgICAgICAgICAgbXRoLk1hdDQucm90YXRlWShzeXN0ZW0udGltZXIudGltZSkuXHJcbiAgICAgICAgICBtdWwobXRoLk1hdDQudHJhbnNsYXRlKG5ldyBtdGguVmVjMyhpICogMi40MCwgMCwgMCkpKVxyXG4gICAgICAgICk7XHJcbiAgICAgIH1cclxuICAgIH1cclxuICB9O1xyXG59IC8qIGNyZWF0ZSAqL1xyXG5cclxuLyogcGxhdG9uaWNfYm9kaWVzLmpzICovIiwiaW1wb3J0ICogYXMgdGNnbCBmcm9tIFwiLi9zeXN0ZW0uanNcIjtcclxuaW1wb3J0ICogYXMgbXRoIGZyb20gXCIuL210aC5qc1wiO1xyXG5pbXBvcnQgKiBhcyBwbGF0b25pY19ib2RpZXMgZnJvbSBcIi4vdW5pdHMvcGxhdG9uaWNfYm9kaWVzLmpzXCI7XHJcblxyXG5sZXQgc3lzdGVtID0gbmV3IHRjZ2wuU3lzdGVtKCk7XHJcblxyXG5kb2N1bWVudC5vbmtleWRvd24gPSAoZXZlbnQpID0+IHtcclxuICBpZiAoZXZlbnQua2V5ICE9ICdQJylcclxuICAgIHJldHVybjtcclxuXHJcbiAgc3lzdGVtLmFkZFVuaXQoYXN5bmMgZnVuY3Rpb24oKSB7XHJcbiAgICBsZXQgdHBsID0gdGNnbC5ybmQuVG9wb2xvZ3kuc3F1YXJlKCk7XHJcbiAgICBsZXQgbXRsID0gYXdhaXQgc3lzdGVtLnJlbmRlci5jcmVhdGVNYXRlcmlhbChcIi4vYmluL3NoYWRlcnMvc2t5c3BoZXJlXCIpO1xyXG5cclxuICAgIGxldCB0ZXggPSBzeXN0ZW0ucmVuZGVyLmNyZWF0ZVRleHR1cmUoKTtcclxuICAgIHRleC5sb2FkKFwiLi9iaW4vc2t5c3BoZXJlcy9za3kxMi5wbmdcIik7XHJcbiAgICBsZXQgZ2wgPSBzeXN0ZW0ucmVuZGVyLmdsO1xyXG4gICAgZ2wudGV4UGFyYW1ldGVyaShnbC5URVhUVVJFXzJELCBnbC5URVhUVUVSX01JTl9GSUxURVIsIGdsLkxJTkVBUl9NSVBNQVBfTElORUFSKTtcclxuICAgIGdsLnRleFBhcmFtZXRlcmkoZ2wuVEVYVFVSRV8yRCwgZ2wuVEVYVFVSRV9NQUdfRklMVEVSLCBnbC5MSU5FQVJfTUlQTUFQX0xJTkVBUik7XHJcblxyXG4gICAgbXRsLnRleHR1cmVzLnB1c2godGV4KTtcclxuICAgIG10bC51Ym8gPSBzeXN0ZW0ucmVuZGVyLmNyZWF0ZVVuaWZvcm1CdWZmZXIoKTtcclxuICAgIG10bC51Ym9OYW1lT25TaGFkZXIgPSBcInByb2plY3Rpb25JbmZvXCI7XHJcblxyXG4gICAgbGV0IHByaW0gPSBhd2FpdCBzeXN0ZW0ucmVuZGVyLmNyZWF0ZVByaW1pdGl2ZSh0cGwsIG10bCk7XHJcblxyXG4gICAgcmV0dXJuIHtcclxuICAgICAgcmVzcG9uc2Uoc3lzdGVtKSB7XHJcbiAgICAgICAgbGV0IGRpciA9IHN5c3RlbS5yZW5kZXIuY2FtZXJhLmRpci5tdWwoc3lzdGVtLnJlbmRlci5jYW1lcmEubmVhcik7XHJcbiAgICAgICAgbGV0IHJnaCA9IHN5c3RlbS5yZW5kZXIuY2FtZXJhLnJpZ2h0Lm11bChzeXN0ZW0ucmVuZGVyLmNhbWVyYS5wcm9qU2l6ZS53KTtcclxuICAgICAgICBsZXQgdHVwID0gc3lzdGVtLnJlbmRlci5jYW1lcmEudXAubXVsKHN5c3RlbS5yZW5kZXIuY2FtZXJhLnByb2pTaXplLmgpO1xyXG4gICAgICAgIFxyXG4gICAgICAgIG10bC51Ym8ud3JpdGVEYXRhKG5ldyBGbG9hdDMyQXJyYXkoW1xyXG4gICAgICAgICAgZGlyLngsIGRpci55LCBkaXIueiwgMCxcclxuICAgICAgICAgIHJnaC54LCByZ2gueSwgcmdoLnosIDAsXHJcbiAgICAgICAgICB0dXAueCwgdHVwLnksIHR1cC56LCAwXHJcbiAgICAgICAgXSkpO1xyXG5cclxuICAgICAgICBzeXN0ZW0ucmVuZGVyLmRyYXdQcmltaXRpdmUocHJpbSk7XHJcbiAgICAgIH1cclxuICAgIH07XHJcbiAgfSk7XHJcbn07XHJcblxyXG5zeXN0ZW0uYWRkVW5pdChwbGF0b25pY19ib2RpZXMuY3JlYXRlKTtcclxuXHJcbnN5c3RlbS5hZGRVbml0KGFzeW5jIGZ1bmN0aW9uKCkge1xyXG4gICAgbGV0IHRwbCA9IGF3YWl0IHRjZ2wucm5kLlRvcG9sb2d5Lm1vZGVsX29iaihcIi4vYmluL21vZGVscy9oZWFkY3JhYi9oZWFkY3JhYi5vYmpcIik7XHJcbiAgICBsZXQgbXRsID0gYXdhaXQgc3lzdGVtLnJlbmRlci5jcmVhdGVNYXRlcmlhbChcIi4vYmluL3NoYWRlcnMvZGVmYXVsdFwiKTtcclxuICAgIGxldCB0ZXggPSBzeXN0ZW0ucmVuZGVyLmNyZWF0ZVRleHR1cmUoKTtcclxuICB0ZXgubG9hZChcIi4vYmluL21vZGVscy9oZWFkY3JhYi9kaWZmdXNlLnBuZ1wiKTtcclxuICBtdGwudGV4dHVyZXMucHVzaCh0ZXgpO1xyXG5cclxuICBsZXQgcHJpbSA9IGF3YWl0IHN5c3RlbS5yZW5kZXIuY3JlYXRlUHJpbWl0aXZlKHRwbCwgbXRsKTtcclxuICBsZXQgdHJhbnNmb3JtID0gbXRoLk1hdDQudHJhbnNsYXRlKG5ldyBtdGguVmVjMyg1LCAwLCAtMTIpKTtcclxuXHJcbiAgcmV0dXJuIHtcclxuICAgIHJlc3BvbnNlKHN5c3RlbSkge1xyXG4gICAgICBzeXN0ZW0ucmVuZGVyLmRyYXdQcmltaXRpdmUocHJpbSwgdHJhbnNmb3JtKTtcclxuICAgIH1cclxuICB9O1xyXG59KTtcclxuXHJcbi8vIGNhbWVyYSB1bml0XHJcbnN5c3RlbS5hZGRVbml0KGZ1bmN0aW9uKCkge1xyXG4gIGNvbnN0IHVwID0gbmV3IG10aC5WZWMzKDAsIDEsIDApO1xyXG4gIGxldCBsb2MgPSBuZXcgbXRoLlZlYzMoMTkuODgsIDExLjY3LCA5LjIwKSwgYXQgPSBuZXcgbXRoLlZlYzMoOC45MCwgNS4zMiwgLTQuNjUpO1xyXG4gIGxldCByYWRpdXMgPSBhdC5zdWIobG9jKS5sZW5ndGgoKTtcclxuXHJcbiAgbGV0IGNhbWVyYSA9IHtcclxuICAgIHJlc3BvbnNlKHN5c3RlbSkge1xyXG4gICAgICBzeXN0ZW0ucmVuZGVyLmNhbWVyYS5zZXQobG9jLCBhdCwgdXApO1xyXG4gICAgfSAvKiByZXNwb25zZSAqL1xyXG4gIH07XHJcblxyXG4gIGNvbnN0IG9uTW91c2VNb3ZlID0gZnVuY3Rpb24oZXZlbnQpIHtcclxuICAgIGlmICgoZXZlbnQuYnV0dG9ucyAmIDEpID09IDEpIHsgLy8gcm90YXRlXHJcbiAgICAgIGxldCBkaXJlY3Rpb24gPSBsb2Muc3ViKGF0KTtcclxuXHJcbiAgICAgIC8vIHR1cm4gZGlyZWN0aW9uIHRvIHBvbGFyIGNvb3JkaW5hdGUgc3lzdGVtXHJcbiAgICAgIHJhZGl1cyA9IGRpcmVjdGlvbi5sZW5ndGgoKTtcclxuICAgICAgbGV0XHJcbiAgICAgICAgYXppbXV0aCAgPSBNYXRoLnNpZ24oZGlyZWN0aW9uLnopICogTWF0aC5hY29zKGRpcmVjdGlvbi54IC8gTWF0aC5zcXJ0KGRpcmVjdGlvbi54ICogZGlyZWN0aW9uLnggKyBkaXJlY3Rpb24ueiAqIGRpcmVjdGlvbi56KSksXHJcbiAgICAgICAgZWxldmF0b3IgPSBNYXRoLmFjb3MoZGlyZWN0aW9uLnkgLyBkaXJlY3Rpb24ubGVuZ3RoKCkpO1xyXG5cclxuICAgICAgLy8gcm90YXRlIGRpcmVjdGlvblxyXG4gICAgICBhemltdXRoICArPSBldmVudC5tb3ZlbWVudFggLyAyMDAuMDtcclxuICAgICAgZWxldmF0b3IgLT0gZXZlbnQubW92ZW1lbnRZIC8gMjAwLjA7XHJcblxyXG4gICAgICBlbGV2YXRvciA9IE1hdGgubWluKE1hdGgubWF4KGVsZXZhdG9yLCAwLjAxKSwgTWF0aC5QSSk7XHJcblxyXG4gICAgICAgLy8gcmVzdG9yZSBkaXJlY3Rpb25cclxuICAgICAgZGlyZWN0aW9uLnggPSByYWRpdXMgKiBNYXRoLnNpbihlbGV2YXRvcikgKiBNYXRoLmNvcyhhemltdXRoKTtcclxuICAgICAgZGlyZWN0aW9uLnkgPSByYWRpdXMgKiBNYXRoLmNvcyhlbGV2YXRvcik7XHJcbiAgICAgIGRpcmVjdGlvbi56ID0gcmFkaXVzICogTWF0aC5zaW4oZWxldmF0b3IpICogTWF0aC5zaW4oYXppbXV0aCk7XHJcblxyXG4gICAgICBsb2MgPSBhdC5hZGQoZGlyZWN0aW9uKTtcclxuICAgIH1cclxuXHJcbiAgICBpZiAoKGV2ZW50LmJ1dHRvbnMgJiAyKSA9PSAyKSB7IC8vIG1vdmVcclxuICAgICAgbGV0IGRpciA9IGF0LnN1Yihsb2MpLm5vcm1hbGl6ZSgpO1xyXG4gICAgICBsZXQgcmdoID0gZGlyLmNyb3NzKHVwKS5ub3JtYWxpemUoKTtcclxuICAgICAgbGV0IHR1cCA9IHJnaC5jcm9zcyhkaXIpO1xyXG5cclxuICAgICAgbGV0IGRlbHRhID0gcmdoLm11bCgtZXZlbnQubW92ZW1lbnRYICogcmFkaXVzIC8gMzAwLjApLmFkZCh0dXAubXVsKGV2ZW50Lm1vdmVtZW50WSAqIHJhZGl1cyAvIDMwMC4wKSk7XHJcbiAgICAgIGxvYyA9IGxvYy5hZGQoZGVsdGEpO1xyXG4gICAgICBhdCA9IGF0LmFkZChkZWx0YSk7XHJcbiAgICB9XHJcbiAgfTtcclxuXHJcbiAgY29uc3Qgb25XaGVlbCA9IGZ1bmN0aW9uKGV2ZW50KSB7XHJcbiAgICBsZXQgZGVsdGEgPSBldmVudC5kZWx0YVkgLyAxMDAuMDtcclxuXHJcbiAgICBsb2MgPSBsb2MuYWRkKGF0LnN1Yihsb2MpLm11bChkZWx0YSAqIHN5c3RlbS50aW1lci5kZWx0YVRpbWUpKTtcclxuICB9O1xyXG5cclxuICBsZXQgY2FudmFzID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoXCJnbENhbnZhc1wiKTtcclxuXHJcbiAgY2FudmFzLmFkZEV2ZW50TGlzdGVuZXIoXCJtb3VzZW1vdmVcIiwgb25Nb3VzZU1vdmUpO1xyXG4gIGNhbnZhcy5hZGRFdmVudExpc3RlbmVyKFwid2hlZWxcIiwgb25XaGVlbCk7XHJcblxyXG4gIHJldHVybiBjYW1lcmE7XHJcbn0pO1xyXG5cclxuLy8gLy8gRlBTIGNvbnRyb2xsZXIgdW5pdFxyXG4vLyBzeXN0ZW0uYWRkVW5pdChmdW5jdGlvbigpIHtcclxuLy8gICBsZXQgdSA9IHtcclxuLy8gICAgIHRleHQ6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKFwiRlBTXCIpLFxyXG4vLyAgICAgcmVzcG9uc2U6IGZ1bmN0aW9uKCkge1xyXG4vLyAgICAgICB1LnRleHQuaW5uZXJUZXh0ID0gYEZQUzogJHtzeXN0ZW0udGltZXIuZnBzfWA7XHJcbi8vICAgICB9XHJcbi8vICAgfTtcclxuXHJcbi8vICAgcmV0dXJuIHU7XHJcbi8vIH0pO1xyXG5cclxuc3lzdGVtLnJ1bigpO1xyXG5cclxuLyogbWFpbi5qcyAqLyJdLCJuYW1lcyI6WyJtdGguU2l6ZSIsIm10aC5WZWMzIiwibXRoLlZlYzIiLCJtdGguQ2FtZXJhIiwibXRoLk1hdDQiLCJybmQuUmVuZGVyIiwidGNnbC5ybmQuVG9wb2xvZ3kiLCJ0Y2dsLnJuZC5WZXJ0ZXgiLCJ0Y2dsLlN5c3RlbSIsInBsYXRvbmljX2JvZGllcy5jcmVhdGUiXSwibWFwcGluZ3MiOiJBQWNBLFNBQVMsc0JBQXNCLENBQUMsRUFBRSxFQUFFLElBQUksRUFBRSxNQUFNLEVBQUU7QUFDbEQsRUFBRSxJQUFJLE1BQU0sSUFBSSxJQUFJLEVBQUU7QUFDdEIsSUFBSSxPQUFPLElBQUksQ0FBQztBQUNoQixHQUFHO0FBQ0g7QUFDQSxFQUFFLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxZQUFZLENBQUMsSUFBSSxDQUFDLENBQUM7QUFDckM7QUFDQSxFQUFFLEVBQUUsQ0FBQyxZQUFZLENBQUMsTUFBTSxFQUFFLE1BQU0sQ0FBQyxDQUFDO0FBQ2xDLEVBQUUsRUFBRSxDQUFDLGFBQWEsQ0FBQyxNQUFNLENBQUMsQ0FBQztBQUMzQjtBQUNBLEVBQUUsSUFBSSxHQUFHLEdBQUcsRUFBRSxDQUFDLGdCQUFnQixDQUFDLE1BQU0sQ0FBQyxDQUFDO0FBQ3hDLEVBQUUsSUFBSSxHQUFHLElBQUksSUFBSSxJQUFJLEdBQUcsQ0FBQyxNQUFNLEdBQUcsQ0FBQztBQUNuQyxJQUFJLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxpQ0FBaUMsRUFBRSxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDN0Q7QUFDQSxFQUFFLE9BQU8sTUFBTSxDQUFDO0FBQ2hCLENBQUM7QUFDRDtBQUNPLFNBQVMsZ0JBQWdCLENBQUMsRUFBRSxFQUFFLFVBQVUsRUFBRSxVQUFVLEVBQUUsVUFBVSxFQUFFO0FBQ3pFLEVBQUUsSUFBSSxPQUFPLEdBQUc7QUFDaEIsSUFBSSxzQkFBc0IsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLGFBQWEsRUFBRSxVQUFVLENBQUM7QUFDNUQsSUFBSSxzQkFBc0IsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLGVBQWUsRUFBRSxVQUFVLENBQUM7QUFDOUQsSUFBSSxzQkFBc0IsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLGVBQWUsRUFBRSxVQUFVLENBQUM7QUFDOUQsR0FBRyxDQUFDO0FBQ0osRUFBRSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsYUFBYSxFQUFFLENBQUM7QUFDbkM7QUFDQSxFQUFFLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUU7QUFDOUIsSUFBSSxJQUFJLE9BQU8sQ0FBQyxDQUFDLENBQUMsSUFBSSxJQUFJLEVBQUU7QUFDNUIsTUFBTSxFQUFFLENBQUMsWUFBWSxDQUFDLE9BQU8sRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUMzQyxLQUFLO0FBQ0wsR0FBRztBQUNIO0FBQ0EsRUFBRSxFQUFFLENBQUMsV0FBVyxDQUFDLE9BQU8sQ0FBQyxDQUFDO0FBQzFCO0FBQ0EsRUFBRSxJQUFJLENBQUMsRUFBRSxDQUFDLG1CQUFtQixDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsV0FBVyxDQUFDO0FBQ3RELElBQUksT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLHNCQUFzQixFQUFFLEVBQUUsQ0FBQyxpQkFBaUIsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUM1RTtBQUNBLEVBQUUsT0FBTyxPQUFPLENBQUM7QUFDakIsQ0FBQztBQUNEO0FBQ08sZUFBZSxVQUFVLENBQUMsRUFBRSxFQUFFLElBQUksRUFBRTtBQUMzQyxFQUFFLE9BQU8sZ0JBQWdCLENBQUMsRUFBRTtBQUM1QixJQUFJLE1BQU0sS0FBSyxDQUFDLElBQUksR0FBRyxRQUFRLEdBQUcsSUFBSSxDQUFDLE1BQU0sRUFBRSxDQUFDLFFBQVEsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLFFBQVEsSUFBSSxRQUFRLENBQUMsRUFBRSxHQUFHLFFBQVEsQ0FBQyxJQUFJLEVBQUUsR0FBRyxJQUFJLENBQUM7QUFDbEgsSUFBSSxNQUFNLEtBQUssQ0FBQyxJQUFJLEdBQUcsUUFBUSxHQUFHLElBQUksQ0FBQyxNQUFNLEVBQUUsQ0FBQyxRQUFRLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxRQUFRLElBQUksUUFBUSxDQUFDLEVBQUUsR0FBRyxRQUFRLENBQUMsSUFBSSxFQUFFLEdBQUcsSUFBSSxDQUFDO0FBQ2xILElBQUksTUFBTSxLQUFLLENBQUMsSUFBSSxHQUFHLFFBQVEsR0FBRyxJQUFJLENBQUMsTUFBTSxFQUFFLENBQUMsUUFBUSxFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsUUFBUSxJQUFJLFFBQVEsQ0FBQyxFQUFFLEdBQUcsUUFBUSxDQUFDLElBQUksRUFBRSxHQUFHLElBQUksQ0FBQztBQUNsSCxHQUFHLENBQUM7QUFDSixDQUFDOztBQzNETSxNQUFNLElBQUksQ0FBQztBQUNsQixFQUFFLENBQUMsQ0FBQztBQUNKLEVBQUUsQ0FBQyxDQUFDO0FBQ0osRUFBRSxDQUFDLENBQUM7QUFDSjtBQUNBLEVBQUUsV0FBVyxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFO0FBQzFCLElBQUksSUFBSSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUM7QUFDaEIsSUFBSSxJQUFJLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQztBQUNoQixJQUFJLElBQUksQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDO0FBQ2hCLEdBQUc7QUFDSDtBQUNBLEVBQUUsSUFBSSxHQUFHO0FBQ1QsSUFBSSxPQUFPLElBQUksSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDNUMsR0FBRztBQUNIO0FBQ0EsRUFBRSxHQUFHLENBQUMsRUFBRSxFQUFFO0FBQ1YsSUFBSSxPQUFPLElBQUksSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDakUsR0FBRztBQUNIO0FBQ0EsRUFBRSxHQUFHLENBQUMsRUFBRSxFQUFFO0FBQ1YsSUFBSSxPQUFPLElBQUksSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDakUsR0FBRztBQUNIO0FBQ0EsRUFBRSxHQUFHLENBQUMsRUFBRSxFQUFFO0FBQ1YsSUFBSSxJQUFJLE9BQU8sRUFBRSxDQUFDLElBQUksTUFBTTtBQUM1QixNQUFNLE9BQU8sSUFBSSxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUNuRTtBQUNBLE1BQU0sT0FBTyxJQUFJLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLEVBQUUsSUFBSSxJQUFJLENBQUMsQ0FBQyxHQUFHLEVBQUUsSUFBSSxJQUFJLENBQUMsQ0FBQyxHQUFHLEVBQUUsR0FBRyxDQUFDO0FBQ25FLEdBQUc7QUFDSDtBQUNBLEVBQUUsTUFBTSxHQUFHO0FBQ1gsSUFBSSxPQUFPLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUMxRSxHQUFHO0FBQ0g7QUFDQSxFQUFFLEdBQUcsQ0FBQyxFQUFFLEVBQUU7QUFDVixJQUFJLE9BQU8sSUFBSSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxD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LE1BQU0sSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUM7QUFDcEQsTUFBTSxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUNwRCxNQUFNLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDO0FBQ3BELEtBQUssQ0FBQztBQUNOLEdBQUc7QUFDSDtBQUNBLEVBQUUsY0FBYyxDQUFDLENBQUM7QUFDbEIsRUFBRTtBQUNGLElBQUksT0FBTyxJQUFJLElBQUk7QUFDbkIsTUFBTSxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUN6RSxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDO0FBQ3pFLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUM7QUFDekUsS0FBSyxDQUFDO0FBQ04sR0FBRztBQUNIO0FBQ0EsRUFBRSxZQUFZLENBQUMsQ0FBQztBQUNoQixFQUFFO0FBQ0YsSUFBSSxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDO0FBQzlFO0FBQ0EsSUFBSSxPQUFPLElBQUksSUFBSTtBQUNuQixNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsSUFBSSxDQUFDO0FBQy9FLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxJQUFJLENBQUM7QUFDL0UsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLElBQUksQ0FBQztBQUMvRSxLQUFLLENBQUM7QUFDTixHQUFHO0FBQ0g7QUFDQSxFQUFFLFNBQVMsR0FBRztBQUNkLElBQUksT0FBTyxJQUFJLElBQUk7QUFDbkIsTUFBTSxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUNwRCxNQUFNLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDO0FBQ3BELE1BQU0sSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUM7QUFDcEQsTUFBTSxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUNwRCxLQUFLLENBQUM7QUFDTixHQUFHO0FBQ0g7QUFDQSxFQUFFLEdBQUcsQ0FBQyxFQUFFLEVBQUU7QUFDVixJQUFJLE9BQU8sSUFBSSxJQUFJO0FBQ25CLE1BQU0sSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUNuRyxNQUFNLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUM7QUFDbkcsTUFBTSxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDO0FBQ25HLE1BQU0sSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUNuRztBQUNBLE1BQU0sSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUNuRyxNQUFNLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUM7QUFDbkcsTUFBTSxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDO0FBQ25HLE1BQU0sSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUNuRztBQUNBLE1BQU0sSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUNuRyxNQUFNLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUM7QUFDbkcsTUFBTSxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDO0FBQ25HLE1BQU0sSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUNuRztBQUNBLE1BQU0sSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQztBQUNuRyxNQUFNLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUM7QUFDbkcsTUFBTSxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDO0FBQ25HLE1BQU0sSUFBSSxDQUFD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Q0FBQyxHQUFHLENBQUMsQ0FBQztBQUN0QyxHQUFHO0FBQ0g7QUFDQSxFQUFFLEdBQUcsQ0FBQyxHQUFHLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRTtBQUNuQixJQUFJLElBQUksQ0FBQyxJQUFJLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDO0FBQ3ZDLElBQUksSUFBSSxDQUFDLFFBQVEsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7QUFDN0M7QUFDQSxJQUFJLElBQUksQ0FBQyxHQUFHLEdBQUcsR0FBRyxDQUFDLElBQUksRUFBRSxDQUFDO0FBQzFCLElBQUksSUFBSSxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUMsSUFBSSxFQUFFLENBQUM7QUFDeEI7QUFDQSxJQUFJLElBQUksQ0FBQyxLQUFLLEdBQUcsSUFBSSxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUM3RSxJQUFJLElBQUksQ0FBQyxFQUFFLE1BQU0sSUFBSSxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUM3RSxJQUFJLElBQUksQ0FBQyxHQUFHLEtBQUssSUFBSSxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLElB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O0FBQ3hGLEVBQUUsQ0FBQyxNQUFNLEVBQUUsc0JBQXNCLENBQUMsMkJBQTJCLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsTUFBTSxDQUFDO0FBQ3hGLEVBQUUsQ0FBQyxNQUFNLEVBQUUsc0JBQXNCLENBQUMsMkJBQTJCLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsTUFBTSxDQUFDO0FBQ3hGLEVBQUUsQ0FBQyxNQUFNLEVBQUUsc0JBQXNCLENBQUMsMkJBQTJCLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsTUFBTSxDQUFDO0FBQ3hGLEVBQUUsQ0FBQyxNQUFNLEVBQUUsc0JBQXNCLENBQUMsMkJBQTJCLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsTUFBTSxDQUFDO0FBQ3hGLEVBQUUsQ0FBQyxNQUFNLEVBQUUsc0JBQXNCLENBQUMsMkJBQTJCLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsTUFBTSxDQUFDO0FBQ3hGLENBQUMsQ0FBQztBQUNGO0FBQ08sTUFBTSxPQUFPLENBQUM7QUFDckIsRUFBRSxHQUFHLENBQUM7QUFDTixFQUFFLEVBQUUsQ0FBQztBQUNMO0FBQ0EsRUFBRSxXQUFXLENBQUMsRUFBRSxFQUFFO0FBQ2xCLElBQUksSUFBSSxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUM7QUFDakI7QUFDQSxJQUFJLElBQUksR0FBRyxHQUFHLFFBQVEsQ0FBQyxhQUFhLENBQUMsUUFBUSxDQUFDLENBQUMsVUFBVSxDQUFDLElBQUksQ0FBQyxDQUFDO0FBQ2hFO0FBQ0EsSUFBSSxHQUFHLENBQUMsTUFBTSxDQUFDLEtBQUssR0FBRyxHQUFHLENBQUM7QUFDM0IsSUFBSSxHQUFHLENBQUMsTUFBTSxDQUFDLE1BQU0sR0FBRyxHQUFHLENBQUM7QUFDNUI7QUFDQSxJQUFJLFNBQVMsUUFBUSxDQUFDLElBQUksRUFBRTtBQUM1QixNQUFNLE1BQU0sQ0FBQyxLQUFLLEVBQUUsTUFBTSxDQUFDLEdBQUcsR0FBRyxD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QUMsR0FBRyxJQUFJLENBQUMsUUFBUSxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUU7QUFDakQsTUFBTSxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsTUFBTSxFQUFFLENBQUMsQ0FBQyxDQUFDO0FBQzVDLEdBQUc7QUFDSCxDQUFDOztBQ3JCTSxNQUFNLE1BQU0sQ0FBQztBQUNwQixFQUFFLENBQUMsQ0FBQztBQUNKLEVBQUUsQ0FBQyxDQUFDO0FBQ0osRUFBRSxDQUFDLENBQUM7QUFDSjtBQUNBLEVBQUUsV0FBVyxDQUFDLFFBQVEsRUFBRSxRQUFRLEVBQUUsTUFBTSxFQUFFO0FBQzFDLElBQUksSUFBSSxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUM7QUFDdEIsSUFBSSxJQUFJLENBQUMsQ0FBQyxHQUFHLFFBQVEsQ0FBQztBQUN0QixJQUFJLElBQUksQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDO0FBQ3BCLEdBQUc7QUFDSDtBQUNBLEVBQUUsT0FBTyxTQUFTLENBQUMsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxHQUFHLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxFQUFFLEdBQUcsR0FBRyxDQUFDLEVBQUUsR0FBRyxHQUFHLENBQUMsRUFBRSxHQUFHLEdBQUcsQ0FBQyxFQUFFO0FBQzFFLElBQUksT0FBTyxJQUFJLE1BQU0sQ0FBQyxJQUFJQyxJQUFRLENBQUMsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxJQUFJQyxJQUFRLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLElBQUlELElBQVEsQ0FBQyxHQUFHLEVBQUUsR0FBRyxFQUFFLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFDbkcsR0FBRztBQUNIO0FBQ0EsRUFBRSxPQUFPLFdBQVcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUlDLElBQVEsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUlELElBQVEsQ0FBQyxD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Q0FBQyxJQUFJLEVBQUUsQ0FBQztBQUN2QjtBQUNBLElBQUksSUFBSSxVQUFVLEdBQUcsSUFBSSxZQUFZLENBQUMsRUFBRSxDQUFDLENBQUM7QUFDMUM7QUFDQSxJQUFJLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUU7QUFDakMsTUFBTSxVQUFVLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxHQUFHLElBQUksQ0FBQyxNQUFNLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUNyRCxLQUFLO0FBQ0wsSUFBSSxVQUFVLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLE1BQU0sQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQ3ZDLElBQUksVUFBVSxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksQ0FBQyxNQUFNLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUN2QyxJQUFJLFVBQVUsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFDdkM7QUFDQSxJQUFJLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLEtBQUssR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sRUFBRSxDQUFDLEdBQUcsS0FBSyxFQUFFLENBQUMsRUFBRSxFQUFFO0FBQ3JFLE1BQU0sSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxTQUFTLENBQUM7QUFDL0MsTUFBTSxJQUFJLEtBQUssR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFDLFNBQVMsQ0FBQztBQUNoRDtBQUNBLE1BQU0sS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRTtBQUNuQyxRQUFRLFVBQVUsQ0FBQyxDQUFDLENBQUMsR0FBRyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0FBQ25DLE9BQU87QUFDUDtBQUNBLE1BQU0sSUFBSSxDQUFDLFNBQVMsQ0FBQyxTQUFTLENBQUMsVUFBVSxDQUFDLENBQUM7QUFDM0M7QUFDQSxNQUFNLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDO0FBQ2hDLEtBQUs7QUFDTDtBQUNBO0FBQ0EsSUFBSSxJQUFJLENBQUMsV0FBVyxHQUFHLEVBQUUsQ0FBQztBQUMxQjtBQUNBO0FBQ0EsSUFBSSxNQUFNLENBQUMsT0FBTyxDQUFDLEVBQUUsQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDO0FBQzlCLElBQUksSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDO0FBQzFDLEdBQUc7QUFDSCxDQUFDO0FBQ0Q7QUFDQTs7QUN6SUEsU0FBUyxPQUFPLEdBQUc7QUFDbkIsRUFBRSxPQUFPLElBQUksQ0FBQyxHQUFHLEVBQUUsR0FBRyxNQUFNLENBQUM7QUFDN0IsQ0FBQztBQUNEO0FBQ08sTUFBTSxLQUFLLENBQUM7QUFDbkIsRUFBRSxpQkFBaUIsR0FBRyxJQUFJLENBQUM7QUFDM0IsRUFBRSxTQUFTLENBQUM7QUFDWixFQUFFLFVBQVUsR0FBRyxJQUFJLENBQUM7QUFDcEIsRUFBRSxjQUFjLEdBQUcsSUFBSSxDQUFDO0FBQ3hCLEVBQUUsUUFBUSxHQUFHLEtBQUssQ0FBQztBQUNuQjtBQUNBLEVBQUUsWUFBWSxHQUFHLElBQUksQ0FBQztBQUN0QixFQUFFLEdBQUcsR0FBRyxTQUFTLENBQUM7QUFDbEI7QUFDQSxFQUFFLElBQUksR0FBRyxJQUFJLENBQUM7QUFDZCxFQUFFLFVBQVUsQ0FBQztBQUNiO0FBQ0EsRUFBRSxTQUFTLEdBQUcsSUFBSSxDQUFDO0FBQ25CLEVBQUUsZUFBZSxHQUFHLElBQUksQ0FBQztBQUN6QjtBQUNBLEVBQUUsV0FBVyxHQUFHO0FBQ2hCLElBQUksSUFBSSxDQUFDLFNBQVMsR0FBRyxPQUFPLEVBQUUsQ0FBQztBQUMvQjtBQUNBLElBQUksSUFBSSxDQUFDLFVBQVUsR0FBRyxJQUFJLENBQUMsU0FBUyxDQUFDO0FBQ3JDLEdBQUc7QUFDSDtBQUNBLEVBQUUsUUFBUSxHQUFHO0FBQ2IsSUFBSSxJQUFJLGFBQWEsR0FBRyxPQUFPLEVBQUUsQ0FBQztBQUNsQztBQUNBLElBQUksSUFBSSxDQUFDLGVBQWUsR0FBRyxhQUFhLEdBQUcsSUFBSSxDQUFDLFVBQVUsQ0FBQztBQUMzRCxJQUFJLElBQUksQ0FBQyxVQUFVLEdBQUcsYUFBYSxDQUFDO0FBQ3BDO0FBQ0EsSUFBSSxJQUFJLElBQUksQ0FBQyxRQUFRLEVBQUU7QUFDdkIsTUFBTSxJQUFJLENBQUMsU0FBUyxHQUFHLElBQUksQ0FBQztBQUM1QixNQUFNLElBQUksQ0FBQyxjQUFjLElBQUksSUFBSSxDQUFDLGVBQWUsQ0FBQztBQUNsRCxLQUFLLE1BQU07QUFDWCxNQUFNLElBQUksQ0FBQyxJQUFJLEdBQUcsSUFBSSxDQUFDLFVBQVUsR0FBRyxJQUFJLENBQUMsU0FBUyxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUM7QUFDekUsTUFBTSxJQUFJLENBQUMsU0FBUyxHQUFHLElBQUksQ0FBQyxlQUFlLENBQUM7QUFDNUMsS0FBSztBQUNMO0FBQ0EsSUFBSSxJQUFJLENBQUMsVUFBVSxFQUFFLENBQUM7QUFDdEIsSUFBSSxJQUFJLElBQUksQ0FBQyxVQUFVLEdBQUcsSUFBSSxDQUFDLGlCQUFpQixJQUFJLElBQUksQ0FBQyxZQUFZLEVBQUU7QUFDdkUsTUFBTSxJQUFJLENBQUMsR0FBRyxHQUFHLElBQUksQ0FBQyxVQUFVLElBQUksSUFBSSxDQUFDLFVBQVUsR0FBRyxJQUFJLENBQUMsaUJBQWlCLENBQUMsQ0FBQztBQUM5RTtBQUNBLE1BQU0sSUFBSSxDQUFDLGlCQUFpQixHQUFHLElBQUksQ0FBQyxVQUFVLENBQUM7QUFDL0MsTUFBTSxJQUFJLENBQUMsVUFBVSxHQUFHLENBQUMsQ0FBQztBQUMxQixLQUFLO0FBQ0wsR0FBRztBQUNILENBQUM7QUFDRDtBQUNBOztBQzdDTyxNQUFNLE1BQU0sQ0FBQztBQUNwQixFQUFFLE1BQU0sQ0FBQztBQUNULEVBQUUsS0FBSyxDQUFDO0FBQ1IsRUFBRSxLQUFLLENBQUM7QUFDUjtBQUNBLEVBQUUsV0FBVyxHQUFHO0FBQ2hCLElBQUksSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJQyxNQUFVLEVBQUUsQ0FBQztBQUNuQyxJQUFJLElBQUksQ0FBQyxLQUFLLEdBQUcsSUFBSSxLQUFLLEVBQUUsQ0FBQztBQUM3QixJQUFJLElBQUksQ0FBQyxLQUFLLEdBQUcsRUFBRSxDQUFDO0FBQ3BCLEdBQUc7QUFDSDtBQUNBLEVBQUUsT0FBTyxTQUFTLEdBQUcsU0FBUyxDQUFDLEVBQUU7QUFDakMsSUFBSSxPQUFPLENBQUMsSUFBSSxPQUFPLENBQUMsQ0FBQyxLQUFLLFFBQVEsSUFBSSxPQUFPLENBQUMsQ0FBQyxJQUFJLEtBQUssVUFBVSxDQUFDO0FBQ3ZFLEdBQUc7QUFDSDtBQUNBLEVBQUUsTUFBTSxPQUFPLENBQUMsY0FBYyxFQUFFO0FBQ2hDLElBQUksSUFBSSxHQUFHLEdBQUcsY0FBYyxDQUFDLElBQUksQ0FBQyxDQUFDO0FBQ25DO0FBQ0EsSUFBSSxJQUFJLE1BQU0sQ0FBQyxTQUFTLENBQUMsR0FBRyxDQUFDLEVBQUU7QUFDL0IsTUFBTSxHQUFHLEdBQUcsTUFBTSxHQUFHLENBQUM7QUFDdEIsS0FBSztBQUNMO0FBQ0EsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQztBQUN6QjtBQUNBLElBQUksT0FBTyxHQUFHLENBQUM7QUFDZixHQUFHO0FBQ0g7QUFDQSxFQUFFLEdBQUcsR0FBRztBQUNSLElBQUksSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDO0FBQ3RCO0FBQ0EsSUFBSSxNQUFNLEdBQUcsR0FBRyxpQkFBaUI7QUFDakMsTUFBTSxNQUFNLENBQUMsS0FBSyxDQUFDLFFBQVEsRUFBRSxDQUFDO0FBQzlCO0FBQ0EsTUFBTSxNQUFNLE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxFQUFFLENBQUM7QUFDbEM7QUFDQSxNQUFNLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLEtBQUssR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDLE1BQU0sRUFBRSxDQUFDLEdBQUcsS0FBSyxFQUFFLENBQUMsRUFBRSxFQUFFO0FBQ25FLFFBQVEsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxRQUFRLENBQUMsTUFBTSxDQUFDLENBQUM7QUFDekMsT0FBTztBQUNQO0FBQ0EsTUFBTSxNQUFNLENBQUMsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDO0FBQzFCO0FBQ0EsTUFBTSxNQUFNLENBQUMscUJBQXFCLENBQUMsR0FBRyxDQUFDLENBQUM7QUFDeEMsS0FBSyxDQUFDO0FBQ04sSUFBSSxNQUFNLENBQUMscUJBQXFCLENBQUMsR0FBRyxDQUFDLENBQUM7QUFDdEMsR0FBRztBQUNILENBQUM7O0FDL0NEO0FBQ0EsTUFBTSxvQkFBb0IsR0FBRyxrQkFBa0IsQ0FBQztBQUNoRDtBQUNBO0FBQ0EsSUFBSSxHQUFHLEdBQUcsSUFBSUMsUUFBaUIsRUFBRSxDQUFDO0FBQ2xDLEdBQUcsQ0FBQyxJQUFJLEdBQUdBLFFBQWlCLENBQUMsS0FBSyxDQUFDO0FBR25DO0FBQ0EsR0FBRyxDQUFDLEdBQUcsR0FBRyxFQUFFLENBQUM7QUFDYixHQUFHLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQ0MsTUFBZSxDQUFDLFNBQVMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDakQsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRTtBQUM1QixFQUFFLEdBQUcsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDQSxNQUFlLENBQUMsV0FBVyxDQUFDTixJQUFRLENBQUMsYUFBYSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsRUFBRSxHQUFHLEdBQUcsRUFBRSxvQkFBb0IsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDaEgsQ0FBQztBQUNELEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUU7QUFDNUIsRUFBRSxHQUFHLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQ00sTUFBZSxDQUFDLFdBQVcsQ0FBQ04sSUFBUSxDQUFDLGFBQWEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEVBQUUsR0FBRyxHQUFHLEdBQUcsSUFBSSxDQUFDLEVBQUUsR0FBRyxHQUFHLEVBQUUsSUFBSSxDQUFDLEVBQUUsR0FBRyxvQkFBb0IsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDMUksQ0FBQztBQUNELEdBQUcsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDTSxNQUFlLENBQUMsU0FBUyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0FBQ2xEO0FBQ0E7QUFDQSxHQUFHLENBQUMsR0FBRyxHQUFHLEVBQUUsQ0FBQztBQUNiLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUU7QUFDNUIsRUFBRSxHQUFHLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUNsQixFQUFFLEdBQUcsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUN0QjtBQUNBLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQ3RCLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUNoQztBQUNBLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUNoQyxFQUFFLEdBQUcsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUN0QjtBQUNBLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQ3RCLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQ3RCO0FBQ0EsRUFBRSxHQUFHLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFDdEIsRUFBRSxHQUFHLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQ2hDO0FBQ0EsRUFBRSxHQUFHLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQztBQUNuQixFQUFFLEdBQUcsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUN0QixDQUFDO0FBQ0Q7QUFDQTtBQUNBLEdBQUcsQ0FBQyxXQUFXLEdBQUcsRUFBRSxDQUFDO0FBQ3JCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUU7QUFDNUIsRUFBRSxHQUFHLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUMxQixFQUFFLEdBQUcsQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUM5QixFQUFFLEdBQUcsQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFDeEM7QUFDQSxFQUFFLEdBQUcsQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUM5QixFQUFFLEdBQUcsQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFDeEMsRUFBRSxHQUFHLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFDOUI7QUFDQSxFQUFFLEdBQUcsQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUM5QixFQUFFLEdBQUcsQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFDeEMsRUFBRSxHQUFHLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQ3hDO0FBQ0EsRUFBRSxHQUFHLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQ3hDLEVBQUUsR0FBRyxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQzlCLEVBQUUsR0FBRyxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUM7QUFDM0IsQ0FBQztBQUNEO0FBQ0E7QUFDQTtBQUNBLElBQUksR0FBRyxHQUFHLElBQUlELFFBQWlCLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDO0FBQ3hDLEdBQUcsQ0FBQyxJQUFJLEdBQUdBLFFBQWlCLENBQUMsS0FBSyxDQUFDO0FBQ25DO0FBQ0EsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEdBQUcsQ0FBQyxXQUFXLENBQUMsTUFBTSxHQUFHLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRTtBQUNyRCxFQUFFLElBQUksRUFBRSxHQUFHLEdBQUcsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLFdBQVcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0FBQ2pELEVBQUUsSUFBSSxFQUFFLEdBQUcsR0FBRyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsV0FBVyxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDakQsRUFBRSxJQUFJLEVBQUUsR0FBRyxHQUFHLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxXQUFXLENBQUMsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUNqRDtBQUNBLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUNDLE1BQWUsQ0FBQyxXQUFXLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7QUFDN0UsQ0FBQztBQUNEO0FBQ0EsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRTtBQUM1QixFQUFFLEdBQUcsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFDMUIsRUFBRSxHQUFHLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQzFCO0FBQ0EsRUFBRSxHQUFHLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQzFCLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUMxQjtBQUNBLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUMxQixFQUFFLEdBQUcsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFDMUI7QUFDQSxFQUFFLEdBQUcsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFDMUIsRUFBRSxHQUFHLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUNwQztBQUNBLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUMxQixFQUFFLEdBQUcsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQ3BDO0FBQ0EsRUFBRSxHQUFHLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQzFCLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFDcEMsQ0FBQztBQUNEO0FBQ0E7QUFDQSxJQUFJLEdBQUcsR0FBRyxJQUFJRCxRQUFpQixFQUFFLENBQUM7QUFDbEMsR0FBRyxDQUFDLElBQUksR0FBR0EsUUFBaUIsQ0FBQyxLQUFLLENBQUM7QUFDbkMsR0FBRyxDQUFDLEdBQUcsR0FBRztBQUNWLEVBQUVDLE1BQWUsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7QUFD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QUUsQ0FBQyxJQUFJLENBQUMsQ0FBQztBQUNuRixFQUFFLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsTUFBTSxFQUFFLENBQUM7QUFDcEM7QUFDQSxFQUFFLElBQUksTUFBTSxHQUFHO0FBQ2YsSUFBSSxRQUFRLENBQUMsTUFBTSxFQUFFO0FBQ3JCLE1BQU0sTUFBTSxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsR0FBRyxDQUFDLEdBQUcsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUM7QUFDNUMsS0FBSztBQUNMLEdBQUcsQ0FBQztBQUNKO0FBQ0EsRUFBRSxNQUFNLFdBQVcsR0FBRyxTQUFTLEtBQUssRUFBRTtBQUN0QyxJQUFJLElBQUksQ0FBQyxLQUFLLENBQUMsT0FBTyxHQUFHLENBQUMsS0FBSyxDQUFDLEVBQUU7QUFDbEMsTUFBTSxJQUFJLFNBQVMsR0FBRyxHQUFHLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxDQUFDO0FBQ2xDO0FBQ0E7QUFDQSxNQUFNLE1BQU0sR0FBRyxTQUFTLENBQUMsTUFBTSxFQUFFLENBQUM7QUFDbEMsTUFBTTtBQUNOLFFBQVEsT0FBTyxJQUFJLElBQUksQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDLEdBQUcsU0FBUyxDQUFDLENBQUMsR0FBRyxTQUFTLENBQUMsQ0FBQyxHQUFHLFNBQVMsQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUNySSxRQUFRLFFBQVEsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDLEdBQUcsU0FBUyxDQUFDLE1BQU0sRUFBRSxDQUFDLENBQUM7QUFDL0Q7QUFDQTtBQUNBLE1BQU0sT0FBTyxLQUFLLEtBQUssQ0FBQyxTQUFTLEdBQUcsS0FBSyxDQUFDO0FBQzFDLE1BQU0sUUFBUSxJQUFJLEtBQUssQ0FBQyxTQUFTLEdBQUcsS0FBSyxDQUFDO0FBQzFDO0FBQ0EsTUFBTSxRQUFRLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLFFBQVEsRUFBRSxJQUFJLENBQUMsRUFBRSxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUM7QUFDN0Q7QUFDQTtBQUNBLE1BQU0sU0FBUyxDQUFDLENBQUMsR0FBRyxNQUFNLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxRQUFRLENBQUMsR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLE9BQU8sQ0FBQyxDQUFDO0FBQ3BFLE1BQU0sU0FBUyxDQUFDLENBQUMsR0FBRyxNQUFNLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxRQUFRLENBQUMsQ0FBQztBQUNoRCxNQUFNLFNBQVMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsUUFBUSxDQUFDLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxPQUFPLENBQUMsQ0FBQztBQUNwRTtBQUNBLE1BQU0sR0FBRyxHQUFHLEVBQUUsQ0FBQyxHQUFHLENBQUMsU0FBUyxDQUFDLENBQUM7QUFDOUIsS0FBSztBQUNMO0FBQ0EsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLE9BQU8sR0FBRyxDQUFDLEtBQUssQ0FBQyxFQUFFO0FBQ2xDLE1BQU0sSUFBSSxHQUFHLEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxTQUFTLEVBQUUsQ0FBQztBQUN4QyxNQUFNLElBQUksR0FBRyxHQUFHLEdBQUcsQ0FBQyxLQUFLLENBQUMsRUFBRSxDQUFDLENBQUMsU0FBUyxFQUFFLENBQUM7QUFDMUMsTUFBTSxJQUFJLEdBQUcsR0FBRyxHQUFHLENBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDO0FBQy9CO0FBQ0EsTUFBTSxJQUFJLEtBQUssR0FBRyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLFNBQVMsR0FBRyxNQUFNLEdBQUcsS0FBSyxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLFNBQVMsR0FBRyxNQUFNLEdBQUcsS0FBSyxDQUFDLENBQUMsQ0FBQztBQUM1RyxNQUFNLEdBQUcsR0FBRyxHQUFHLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDO0FBQzNCLE1BQU0sRUFBRSxHQUFHLEVBQUUsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUM7QUFDekIsS0FBSztBQUNMLEdBQUcsQ0FBQztBQUNKO0FBQ0EsRUFBRSxNQUFNLE9BQU8sR0FBRyxTQUFTLEtBQUssRUFBRTtBQUNsQyxJQUFJLElBQUksS0FBSyxHQUFHLEtBQUssQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO0FBQ3JDO0FBQ0EsSUFBSSxHQUFHLEdBQUcsR0FBRyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxLQUFLLEdBQUcsTUFBTSxDQUFDLEtBQUssQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDO0FBQ25FLEdBQUcsQ0FBQztBQUNKO0FBQ0EsRUFBRSxJQUFJLE1BQU0sR0FBRyxRQUFRLENBQUMsY0FBYyxDQUFDLFVBQVUsQ0FBQyxDQUFDO0FBQ25EO0FBQ0EsRUFBRSxNQUFNLENBQUMsZ0JBQWdCLENBQUMsV0FBVyxFQUFFLFdBQVcsQ0FBQyxDQUFDO0FBQ3BELEVBQUUsTUFBTSxDQUFDLGdCQUFnQixDQUFDLE9BQU8sRUFBRSxPQUFPLENBQUMsQ0FBQztBQUM1QztBQUNBLEVBQUUsT0FBTyxNQUFNLENBQUM7QUFDaEIsQ0FBQyxDQUFDLENBQUM7QUFDSDtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQztBQUNiO0FBQ0EifQ==

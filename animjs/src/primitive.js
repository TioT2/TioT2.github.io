import {loadShader} from "./shader.js";
import {Material, Texture, Cubemap, UBO} from "./material.js";
import * as mth from "./mth.js";

export {loadShader, Material, Texture, Cubemap, UBO, mth};

export class Vertex {
  p;
  t;
  n;

  constructor(position, texcoord, normal) {
    this.p = position;
    this.t = texcoord;
    this.n = normal;
  } /* constructor */

  static fromCoord(px, py, pz, pu = 0, pv = 0, pnx = 0, pny = 0, pnz = 1) {
    return new Vertex(new mth.Vec3(px, py, pz), new mth.Vec2(pu, pv), new mth.Vec3(pnx, pny, pnz));
  } /* fromCoord */

  static fromVectors(p, t = new mth.Vec2(0, 0), n = new mth.Vec3(1, 1, 1)) {
    return new Vertex(p.copy(), t.copy(), n.copy());
  } /* fromVectors */
}; /* Vertex */

export class Topology {
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
          positions.push(new mth.Vec3(
            parseFloat(segments[1]),
            parseFloat(segments[2]),
            parseFloat(segments[3])
          ));
          break;

        case "vt":
          texCoords.push(new mth.Vec2(
            parseFloat(segments[1]),
            parseFloat(segments[2])
          ));
          break;

        case "vn":
          normals.push(new mth.Vec3(
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
              Number.isNaN(i0) ? new mth.Vec3(0, 0, 0) : positions[i0 - 1],
              Number.isNaN(i1) ? new mth.Vec2(0, 0) : texCoords[i1 - 1],
              Number.isNaN(i2) ? new mth.Vec3(0, 0, 0) : normals[i2 - 1]
            ));
          }
          {
            let vtd = segments[2].split("/");
            let i0 = parseInt(vtd[0]), i1 = parseInt(vtd[1]), i2 = parseInt(vtd[2]);

            tpl.vtx.push(new Vertex(
              Number.isNaN(i0) ? new mth.Vec3(0, 0, 0) : positions[i0 - 1],
              Number.isNaN(i1) ? new mth.Vec2(0, 0) : texCoords[i1 - 1],
              Number.isNaN(i2) ? new mth.Vec3(0, 0, 0) : normals[i2 - 1]
            ));
          }
          {
            let vtd = segments[3].split("/");
            let i0 = parseInt(vtd[0]), i1 = parseInt(vtd[1]), i2 = parseInt(vtd[2]);

            tpl.vtx.push(new Vertex(
              Number.isNaN(i0) ? new mth.Vec3(0, 0, 0) : positions[i0 - 1],
              Number.isNaN(i1) ? new mth.Vec2(0, 0) : texCoords[i1 - 1],
              Number.isNaN(i2) ? new mth.Vec3(0, 0, 0) : normals[i2 - 1]
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

export class Primitive {
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
}; /* Primitive */

import {Primitive, Topology, Vertex} from "./primitive.js";
import * as mth from "./mth.js";
import {Material, Texture, UBO, loadShader} from "./material.js";

export {Primitive, Topology, Vertex, Texture, UBO, mth};

export class Render {
  renderQueue;
  gl;
  camera;
  cameraUBO;
  testTexture;

  constructor() {
    // WebGL initialization
    let canvas = document.getElementById("glCanvas");
    let gl = canvas.getContext("webgl2");

    this.gl = gl;

    gl.enable(WebGL2RenderingContext.DEPTH_TEST);

    this.renderQueue = [];
    this.camera = new mth.Camera();

    this.cameraUBO = new UBO(this.gl);
    this.testTexture = new Texture(this.gl, "./models/headcrab/diffuse.png");

  } /* constructor */

  drawPrimitive(primitive, transform = mth.Mat4.identity()) {
    this.renderQueue.push({primitive, transform});
  } /* drawPrimitive */

  createTexture(texturePath) {
    return new Texture(this.gl, texturePath);
  } /* createTexture */

  async createMaterial(shaderPath) {
    return new Material(this.gl, await loadShader(this.gl, shaderPath));
  } /* createMaterial */

  createPrimitive(topology, material) {
    return Primitive.fromTopology(this.gl, topology, material);
  } /* createPrimitive */

  start() {
    let gl = this.gl;

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.clearColor(0.30, 0.47, 0.80, 1);
  } /* start */

  end() {
    let gl = this.gl;

    for (let i = 0, count = this.renderQueue.length; i < count; i++) {
      let prim = this.renderQueue[i].primitive;
      let trans = this.renderQueue[i].transform;

      this.cameraUBO.writeData(new Float32Array(trans.m.concat(this.camera.viewProj.m)));

      prim.draw(this.cameraUBO);
    }

    // flush render queue
    this.renderQueue = [];

    gl.finish();
  } /* end */
} /* Render */

/* render.js */
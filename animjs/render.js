import {loadShader, Material, Primitive, Topology, Vertex, Texture, UBO, mth} from "./primitive.js";
import {Target} from "./target.js";

export {Material, Primitive, Topology, Vertex, Texture, UBO, mth};

export class Render {
  renderQueue;
  gl;
  camera;
  cameraUBO;

  target;
  fsPrimitive = null;

  constructor() {
    // WebGL initialization
    let canvas = document.getElementById("glCanvas");
    let gl = canvas.getContext("webgl2");

    this.gl = gl;

    gl.enable(WebGL2RenderingContext.DEPTH_TEST);

    this.renderQueue = [];
    this.camera = new mth.Camera();

    this.cameraUBO = new UBO(this.gl);

    this.camera.resize(new mth.Size(canvas.width, canvas.height));
    gl.viewport(0, 0, canvas.width, canvas.height);

    // targets setup
    let size = new mth.Size(800, 600);
    this.target = new Target(gl, 3);

    Target.default(gl).resize(size);
  } /* constructor */

  drawPrimitive(primitive, transform = mth.Mat4.identity()) {
    this.renderQueue.push({primitive, transform});
  } /* drawPrimitive */

  createTexture() {
    return new Texture(this.gl, Texture.UNSIGNED_BYTE, 4);
  } /* createTexture */

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
      this.fsPrimitive = await this.createPrimitive(Topology.square(), await this.createMaterial("./shaders/target"));
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
    this.fsPrimitive.draw();
  } /* end */
} /* Render */

/* render.js */
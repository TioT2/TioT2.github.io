import {loadShader} from "./shader.js";
import {Texture} from "./texture.js";
import {UBO} from "./ubo.js";
export {Texture, UBO, loadShader};

export class Material {
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
      this.textures[i].bind(i);
  } /* apply */
} /* Material */
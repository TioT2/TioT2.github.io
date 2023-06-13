export class UBO {
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

export class Texture {
  gl;
  id;

  constructor(gl, imagePath) {
    this.gl = gl;
    this.id = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.id);
    
    // put empty image data
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 255, 0, 255]));

    // async image loading
    let image = new Image();
    image.src = imagePath;
    image.addEventListener("load", () => {
      gl.bindTexture(gl.TEXTURE_2D, this.id);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
      gl.generateMipmap(gl.TEXTURE_2D);
    });
  } /* constructor */

  bind(index) {
    this.gl.activeTexture(this.gl.TEXTURE0 + index);
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.id);
  } /* bind */
} /* Texture */
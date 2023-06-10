import * as mth from "./mth.js";
import {Texture} from "./texture.js";

function decodeFramebufferStatus(status) {
  switch (status) {
    case WebGL2RenderingContext.FRAMEBUFFER_COMPLETE:                      return "complete";
    case WebGL2RenderingContext.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:         return "incomplete attachment";
    case WebGL2RenderingContext.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:         return "height and width of attachment aren't same";
    case WebGL2RenderingContext.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: return "attachment missing";
    case WebGL2RenderingContext.FRAMEBUFFER_UNSUPPORTED:                   return "attachment format isn't supported";
  }
}

export class Target {
  #gl;
  FBO;
  attachments = [];
  size;
  depth;

  constructor(gl, attachmentCount) {
    this.size = new mth.Size(800, 600);
    this.gl = gl;
    this.FBO = gl.createFramebuffer();

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.FBO);

    
    // create target textures
    let drawBuffers = [];
    for (let i = 0; i < attachmentCount; i++) {
      this.attachments[i] = new Texture(gl, Texture.UNSIGNED_BYTE, 4);
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

    // console.log(`Framebuffer status: ${decodeFramebufferStatus(gl.checkFramebufferStatus(gl.FRAMEBUFFER))}`);
  } /* constructor */

  resize(size) {
    throw Error("no Implementation");
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
    size: new mth.Size(800, 600),
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
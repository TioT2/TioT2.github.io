import * as camera from "./camera.js"
import * as sets from "./sets.js";

let gl = null;
let canvas = null;

let posBuf = null;

export async function render() {
  if(gl == null || sets.current == null)
    return;

  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.clearColor(0, 1, 0, 0);

  gl.useProgram(sets.current.shader);
  sets.current.onResponse(true);

  // camera params
  let positionLocation = gl.getUniformLocation(sets.current.shader, "Position");
  if (positionLocation != null)
    gl.uniform2f(positionLocation, camera.position.x, camera.position.y);

  let scaleLocation = gl.getUniformLocation(sets.current.shader, "Scale");
  if (scaleLocation != null)
    gl.uniform1f(scaleLocation, camera.scale);

  // set color
  let colorLocation = gl.getUniformLocation(sets.current.shader, "Color");
  if (colorLocation != null) {
    let picker = document.getElementById("setColorPicker");
    let colorCoefficent = document.getElementById("setColorCoefficent");
    let color = parseInt(picker.value.slice(1, 7), 16);
    let rgb = {r: (color >> 16) & 0xFF, g: (color >> 8) & 0xFF, b: (color >> 0) & 0xFF};
    
    gl.uniform3f(colorLocation, colorCoefficent.value * rgb.r / 255.0, colorCoefficent.value * rgb.g / 255.0, colorCoefficent.value * rgb.b / 255.0);
  }
  let projectionLocation = gl.getUniformLocation(sets.current.shader, "Projection");
  if (projectionLocation != null) {
    let wp = 1, hp = 1;

    if (canvas.width > canvas.height) {
      wp *= canvas.width / canvas.height;
    } else {
      hp *= canvas.height / canvas.width;
    }

    gl.uniform2f(projectionLocation, wp, hp);
  }

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  gl.finish();

  window.requestAnimationFrame(render);
} /* render */

async function initPrimitive() {
  // create fullscreen primitive
  const posLoc = gl.getAttribLocation(sets.current.shader, "inPos");
  const pos = [
    -1, -1, 0, 1,
    -1,  1, 0, 1,
    1, -1, 0, 1,
    1,  1, 0, 1,
  ];
  posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.STATIC_DRAW);
  gl.vertexAttribPointer(posLoc, 4, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(posLoc);
} /* initPrimitive */

// WebGL initialization
canvas = document.getElementById("glCanvas");
gl = canvas.getContext("webgl2");

await sets.init(gl);
sets.bind("Mandelbrot");
await initPrimitive();

// configure set selector
let select = document.getElementById("setTypeSelector");
let heading = document.getElementById("heading");
select.addEventListener("change", (Element, Event) => {
  sets.bind(select.value);
  heading.innerHTML = `The ${select.value} set`;
});

camera.init(canvas);

window.requestAnimationFrame(render);

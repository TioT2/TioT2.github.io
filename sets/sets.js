import {loadShader} from "./shader.js"

let gl;

async function juliaSet() {
  let set = {
    real            : null, // real part slider
    imaginary       : null, // imaginary part slider
    coefficentBlock : null, // imaginary setting block

    shader          : null, // set shader
    oldInnerHTML    : "",   // 

    onResponse : function() {
      // send 'C' parameter for Julia set equasion
      let paramLocation = gl.getUniformLocation(this.shader, "CParam");
      if (paramLocation != null)
        gl.uniform2f(paramLocation, this.real.value, this.imaginary.value);
    },

    onApply : function(active) {
      [this.coefficentBlock.innerHTML, this.oldInnerHTML] = [this.oldInnerHTML, this.coefficentBlock.innerHTML];
      if (active) {
        const fields = ["real", "imaginary"];
        fields.forEach(field => set[field] = document.getElementById(field));
      }
    }
  };

  set.shader = await loadShader(gl, "./shaders/sets/julia");
  set.coefficentBlock = document.getElementById("juliaCoefficents");
  [set.coefficentBlock.innerHTML, set.oldInnerHTML] = [set.oldInnerHTML, set.coefficentBlock.innerHTML];

  return set;
} /* juliaSet */

async function mandelbrotSet() {
  let set = {
    shader : null,

    onResponse : function() {
    },

    onApply : function(active) {
    }
  };

  set.shader = await loadShader(gl, "./shaders/sets/mandelbrot");

  return set;
} /* mandelbrotSet */

let sets = {};      // object that contains all sets
export let current = null; // active set

export function bind(name) {
  let newCurrentSet = sets[name];

  if (newCurrentSet != undefined) {
    current.onApply(false);
    newCurrentSet.onApply(true);
    current = newCurrentSet;
  } else {
    throw Error(`Set ${select.value} doesn't exist`);
  }
}

export async function init(glctx) {
  gl = glctx;
  // create sets
  sets = {
    "Julia" : await juliaSet(),
    "Mandelbrot" : await mandelbrotSet()
  };
  current = sets["Mandelbrot"];
  current.onApply(true);
} /* initSets */
export let position = {x: 0, y: 0};
let raw_scale = 0.0;
export let scale = 1.0;

let canvas;

function clamp(value, min, max) {
  return Math.max(Math.min(value, max), min);
}

function onMouseMove(event) {
  if ((event.buttons & 1) == 1)
  {
    let delta = {x: event.movementX, y: event.movementY};

    position.x -= delta.x / canvas.width / scale * 3.0;
    position.y += delta.y / canvas.height / scale * 3.0;

    
    const radius = Math.sqrt(position.x * position.x + position.y * position.y);
    if (radius > 2) {
      let radialCoefficent = Math.min(radius, 2.0) / radius;
      position.x *= radialCoefficent;
      position.y *= radialCoefficent;
    }
  }
} /* onMouseMove */

function onWheel(event) {
  raw_scale = raw_scale + event.deltaY / 1000.0;
  raw_scale = Math.max(raw_scale, 0);
  scale = Math.pow(8, raw_scale);

  event.preventDefault();
} /* onWheel */

export function init(canvas_for) {
  canvas = canvas_for;
  
  // configure input
  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("wheel", onWheel);
}
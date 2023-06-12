import * as tcgl from "./system.js";
import * as mth from "./mth.js";
import * as platonic_bodies from "./units/platonic_bodies.js";

let system = new tcgl.System();

system.addUnit(platonic_bodies.create);

system.addUnit(async function() {
  let tpl = await tcgl.rnd.Topology.model_obj("./models/headcrab/headcrab.obj");
  let mtl = await system.render.createMaterial("./shaders/default");
  let tex = system.render.createTexture();
  tex.load("./models/headcrab/diffuse.png");
  mtl.textures.push(tex);

  let prim = await system.render.createPrimitive(tpl, mtl);

  // let mdl = new Model(system.render.gl);
  // await mdl.loadGLTF("./models/telephone");
  let transform = mth.Mat4.translate(new mth.Vec3(5, 0, -12));

  return {
    response(system) {
      system.render.drawPrimitive(prim, transform);
      // system.render.drawPrimitive(mdl.primitives[0]);
    }
  };
});

// camera unit
system.addUnit(function() {
  const up = new mth.Vec3(0, 1, 0);
  let loc = new mth.Vec3(19.88, 11.67, 9.20), at = new mth.Vec3(8.90, 5.32, -4.65);
  let radius = at.sub(loc).length();

  let camera = {
    response(system) {
      system.render.camera.set(loc, at, up);
    } /* response */
  };

  const onMouseMove = function(event) {
    if ((event.buttons & 1) == 1) { // rotate
      let direction = loc.sub(at);

      // turn direction to polar coordinate system
      radius = direction.length();
      let
        azimuth  = Math.sign(direction.z) * Math.acos(direction.x / Math.sqrt(direction.x * direction.x + direction.z * direction.z)),
        elevator = Math.acos(direction.y / direction.length());

      // rotate direction
      azimuth  += event.movementX / 200.0;
      elevator -= event.movementY / 200.0;

      elevator = Math.min(Math.max(elevator, 0.01), Math.PI);

       // restore direction
      direction.x = radius * Math.sin(elevator) * Math.cos(azimuth);
      direction.y = radius * Math.cos(elevator);
      direction.z = radius * Math.sin(elevator) * Math.sin(azimuth);

      loc = at.add(direction);
    }

    if ((event.buttons & 2) == 2) { // move
      let dir = at.sub(loc).normalize();
      let rgh = dir.cross(up).normalize();
      let tup = rgh.cross(dir);

      let delta = rgh.mul(-event.movementX * radius / 300.0).add(tup.mul(event.movementY * radius / 300.0));
      loc = loc.add(delta);
      at = at.add(delta);
    }
  };

  const onWheel = function(event) {
    let delta = event.deltaY / 100.0;

    loc = loc.add(at.sub(loc).mul(delta * system.timer.deltaTime));
  };

  let canvas = document.getElementById("glCanvas");

  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("wheel", onWheel);

  return camera;
});

// // FPS controller unit
// system.addUnit(function() {
//   let u = {
//     text: document.getElementById("FPS"),
//     response: function() {
//       u.text.innerText = `FPS: ${system.timer.fps}`;
//     }
//   };

//   return u;
// });

system.run();

/* main.js */
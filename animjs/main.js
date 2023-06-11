import * as tcgl from "./system.js";
import * as mth from "./mth.js";
import * as platonic_bodies from "./units/platonic_bodies.js";

let system = new tcgl.System();

// platonic bodies unit

system.addUnit(async function() {
  let tpl = await tcgl.rnd.Topology.sphere();
  let shd = await system.render.createShader("./shaders/default_pbr");

  let prims = [];

  let kds = [
    new mth.Vec3(0.47, 0.78, 0.73),
    new mth.Vec3(0.86, 0.18, 0.00),
    new mth.Vec3(0.01, 0.01, 0.01),
  ];

  for (let fi = 0; fi < kds.length; fi++) {
    for (let i = 0; i < 7; i++) {
      let mtl = await system.render.createMaterial(shd);
      let prim = await system.render.createPrimitive(tpl, mtl);
      mtl.ubo = system.render.createUniformBuffer();

      mtl.ubo.writeData(new Float32Array([
        kds[fi].x, kds[fi].y, kds[fi].z,
        i / 7,
        i / 7 + 0.05,
      ]));
      mtl.uboNameOnShader = "materialUBO";

      prims.push({
        primitive: prim,
        transform: mth.Mat4.translate(new mth.Vec3((i - 3.0) * 2.40, (kds.length / 2 - fi) * 2.40, 0))
      });
    }
  }

  return {
    response(system) {
      for (let i = 0; i < prims.length; i++) {
        system.render.drawPrimitive(prims[i].primitive, prims[i].transform);
      }
    }
  };
});

// system.addUnit(async function() {
//   let tpl = await tcgl.rnd.Topology.model_obj("./models/headcrab/headcrab.obj");
//   let mtl = await system.render.createMaterial("./shaders/default");
//   let tex = system.render.createTexture();
//   tex.load("./models/headcrab/diffuse.png");
//   mtl.textures.push(tex);

//   let prim = await system.render.createPrimitive(tpl, mtl);

//   let mdl = new Model(system.render.gl);
//   await mdl.loadGLTF("./models/telephone");

//   return {
//     response(system) {
//       system.render.drawPrimitive(prim);
//       system.render.drawPrimitive(mdl.primitives[0]);
//     }
//   };
// });

// camera unit
system.addUnit(function() {
  const up = new mth.Vec3(0, 1, 0);
  // let loc = new mth.Vec3(-1.8, 2.1, 4.7), at = new mth.Vec3(4.66, -2, -2.40);
  let loc = new mth.Vec3(0, 0, 20), at = new mth.Vec3(0, 0, 0);
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
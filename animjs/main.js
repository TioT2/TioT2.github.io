import * as tcgl from "./system.js";
import * as mth from "./mth.js";

let system = new tcgl.System();

system.render.camera.resize(new mth.Size(800, 600));

// // add headcrab cube
// system.addUnit(async function() {
//   let mtl = await system.render.createMaterial("./shaders/default");
//   mtl.textures.push(system.render.createTexture("./models/headcrab/diffuse.png"));
//   let prim = await system.render.createPrimitive(await tcgl.rnd.Topology.model_obj("./models/headcrab/headcrab.obj"), mtl);
//   return {
//     response(system) {
//       // let time = system.timer.time;
//       // system.render.camera.set(new mth.Vec3(Math.cos(time), 0.5, Math.sin(time)).mul(700), new mth.Vec3(0, 0, 0), new mth.Vec3(0, 1, 0));
//       const w = 6, h = 6, d = 6;
//       for (let z = -d; z <= d; z++)
//         for (let y = -h; y <= h; y++)
//           for (let x = -w; x <= w; x++)
//             system.render.drawPrimitive(prim, mth.Mat4.translate(new mth.Vec3(x * 16, y * 16, z * 16)));
//     } /* response */
//   };
// });

function writeNormals(vtx, idx) {
  // write normals
  for (let i = 0; i < vtx.length; i++) {
    vtx[i].n = new mth.Vec3(0, 0, 0);
  }
  for (let i = 0; i < idx.length / 3; i++) {
    let v1 = vtx[idx[i * 3 + 0]];
    let v2 = vtx[idx[i * 3 + 1]];
    let v3 = vtx[idx[i * 3 + 2]];
    
    let n = v1.p.sub(v2.p).cross(v3.p.sub(v2.p)).normalize();

    v1.n = v1.n.add(n);
    v2.n = v2.n.add(n);
    v3.n = v3.n.add(n);
  }
  for (let i = 0; i < vtx.length; i++) {
    vtx[i].n = vtx[i].n.normalize();
  }
}

// elevation of icosahedron basic sphere
const icosahedronElevation = 1.1071487177940904;

// icosahedron evaluation
let ico = new tcgl.rnd.Topology();
ico.type = tcgl.rnd.Topology.LINES;

tcgl.rnd.Vertex.fromCoord(0, 1, 0);

ico.vtx = [];
ico.vtx.push(tcgl.rnd.Vertex.fromCoord(0, 1, 0));
for (let a = 0; a < 5; a++) {
  ico.vtx.push(tcgl.rnd.Vertex.fromVectors(mth.Vec3.fromSpherical(a * Math.PI * 0.4, icosahedronElevation, 1)));
}
for (let a = 0; a < 5; a++) {
  ico.vtx.push(tcgl.rnd.Vertex.fromVectors(mth.Vec3.fromSpherical(a * Math.PI * 0.4 + Math.PI / 5.0, Math.PI - icosahedronElevation, 1)));
}
ico.vtx.push(tcgl.rnd.Vertex.fromCoord(0, -1, 0));

// write wire indices
ico.idx = [];
for (let a = 0; a < 5; a++) {
  ico.idx.push(0);
  ico.idx.push(a + 1);

  ico.idx.push(a + 1);
  ico.idx.push((a + 1) % 5 + 1);

  ico.idx.push((a + 1) % 5 + 1);
  ico.idx.push(a + 6);

  ico.idx.push(a + 1);
  ico.idx.push(a + 6);

  ico.idx.push(a + 6);
  ico.idx.push((a + 1) % 5 + 6);

  ico.idx.push(11);
  ico.idx.push(a + 6);
}

// write polygonal indices
ico.polygon_idx = [];
for (let y = 0; y < 5; y++) {
  ico.polygon_idx.push(0);
  ico.polygon_idx.push(y + 1);
  ico.polygon_idx.push((y + 1) % 5 + 1);

  ico.polygon_idx.push(y + 1);
  ico.polygon_idx.push((y + 1) % 5 + 1);
  ico.polygon_idx.push(y + 6);

  ico.polygon_idx.push(y + 6);
  ico.polygon_idx.push((y + 1) % 5 + 1);
  ico.polygon_idx.push((y + 1) % 5 + 6);

  ico.polygon_idx.push((y + 1) % 5 + 6);
  ico.polygon_idx.push(y + 6);
  ico.polygon_idx.push(11);
}


// dodecahedron evaluation
let dod = new tcgl.rnd.Topology([], []);
dod.type = tcgl.rnd.Topology.LINES;

for (let i = 0; i < ico.polygon_idx.length / 3; i++) {
  let p1 = ico.vtx[ico.polygon_idx[i * 3 + 0]].p;
  let p2 = ico.vtx[ico.polygon_idx[i * 3 + 1]].p;
  let p3 = ico.vtx[ico.polygon_idx[i * 3 + 2]].p;

  dod.vtx.push(tcgl.rnd.Vertex.fromVectors(p1.add(p2).add(p3).mul(1.0 / 3)));
}

for (let a = 0; a < 5; a++) {
  dod.idx.push(a * 4 + 0);
  dod.idx.push(a * 4 + 1);

  dod.idx.push(a * 4 + 1);
  dod.idx.push(a * 4 + 2);

  dod.idx.push(a * 4 + 2);
  dod.idx.push(a * 4 + 3);

  dod.idx.push(a * 4 + 0);
  dod.idx.push((a + 1) % 5 * 4 + 0);

  dod.idx.push(a * 4 + 2);
  dod.idx.push((a + 1) % 5 * 4 + 1);

  dod.idx.push(a * 4 + 3);
  dod.idx.push((a + 1) % 5 * 4 + 3);
}

// cube evaluation
let cub = new tcgl.rnd.Topology();
cub.type = tcgl.rnd.Topology.LINES;
cub.vtx = [
  tcgl.rnd.Vertex.fromCoord(-1, -1, -1),
  tcgl.rnd.Vertex.fromCoord(-1,  1, -1),
  tcgl.rnd.Vertex.fromCoord( 1,  1, -1),
  tcgl.rnd.Vertex.fromCoord( 1, -1, -1),

  tcgl.rnd.Vertex.fromCoord(-1, -1,  1),
  tcgl.rnd.Vertex.fromCoord(-1,  1,  1),
  tcgl.rnd.Vertex.fromCoord( 1,  1,  1),
  tcgl.rnd.Vertex.fromCoord( 1, -1,  1),
];

cub.idx = [
  0, 1, 1, 2, 2, 3, 3, 0,
  4, 5, 5, 6, 6, 7, 7, 4,
  0, 4, 1, 5, 2, 6, 3, 7
];

// octahedron evaluation
let oct = new tcgl.rnd.Topology();
oct.type = tcgl.rnd.Topology.LINES;

oct.vtx = [
  tcgl.rnd.Vertex.fromCoord( 0,  1,  0),

  tcgl.rnd.Vertex.fromCoord( 1,  0,  0),
  tcgl.rnd.Vertex.fromCoord( 0,  0,  1),
  tcgl.rnd.Vertex.fromCoord(-1,  0,  0),
  tcgl.rnd.Vertex.fromCoord( 0,  0, -1),

  tcgl.rnd.Vertex.fromCoord( 0, -1,  0),
];
oct.idx = [
  0, 1, 0, 2, 0, 3, 0, 4,
  1, 2, 2, 3, 3, 4, 4, 1,
  5, 1, 5, 2, 5, 3, 5, 4
];

// tetrahedron evalutation
const tetHeight = Math.sqrt(2 / 3);
const tetInnerLen = 1 / Math.sqrt(3);

let tet = new tcgl.rnd.Topology();
tet.type = tcgl.rnd.Topology.LINES;
tet.vtx = [tcgl.rnd.Vertex.fromVectors(new mth.Vec3(0, tetHeight, 0))];
for (let a = 0; a < 3; a++) {
  tet.vtx.push(tcgl.rnd.Vertex.fromVectors(new mth.Vec3(
    Math.cos(2 * Math.PI / 3 * a) * tetInnerLen * 2,
    -tetHeight,
    Math.sin(2 * Math.PI / 3 * a) * tetInnerLen * 2
  )));
}
tet.idx = [0, 1, 0, 2, 0, 3, 1, 2, 2, 3, 3, 1];

system.addUnit(async function() {
  let platonicMtl = await system.render.createMaterial("./shaders/platonic");

  let prims = [
    await system.render.createPrimitive(ico, platonicMtl),
    await system.render.createPrimitive(dod, platonicMtl),
    await system.render.createPrimitive(cub, platonicMtl),
    await system.render.createPrimitive(oct, platonicMtl),
    await system.render.createPrimitive(tet, platonicMtl)
  ];

  return {
    response(system) {
      for (let i = 0; i < prims.length; i++) {
        system.render.drawPrimitive(prims[i],
              mth.Mat4.rotateY(system.timer.time).
          mul(mth.Mat4.translate(new mth.Vec3(i * 2.40, 0, 0)))
        );
      }
    }
  };
});

// camera unit
system.addUnit(function() {
  const up = new mth.Vec3(0, 1, 0);
  let loc = new mth.Vec3(5, 5, 5), at = new mth.Vec3(0, 0, 0);
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

// FPS controller unit
system.addUnit(function() {
  let u = {
    text: document.getElementById("FPS"),
    response: function() {
      u.text.innerText = `FPS: ${system.timer.fps}`;
    }
  };

  return u;
});

system.run();
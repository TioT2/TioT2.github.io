import * as tcgl from "./../system.js";
import * as mth from "./../mth.js";

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

// add platonic bodies units
export async function create(system) {
  let platonicMtl = await system.render.createMaterial("./bin/shaders/platonic");

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
} /* create */

/* platonic_bodies.js */
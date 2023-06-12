import * as tcgl from "./../system.js";
import * as mth from "./../mth.js";

export async function create(system) {
  let tpl = await tcgl.rnd.Topology.sphere();
  let shd = await system.render.createShader("./shaders/pbr_test");

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
}

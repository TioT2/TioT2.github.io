import * as rnd from "./render.js";
import {Timer} from "./timer.js";

export {rnd};

export class System {
  render; // Render
  units;  // unit list
  timer;  // timer

  constructor() {
    this.render = new rnd.Render();
    this.timer = new Timer();
    this.units = [];
  } /* constructor */

  static isPromise = function(v) {
    return v => typeof(v) === "object" && typeof v.then === "function";
  } /* isPromise */

  async addUnit(createFunction) {
    let val = createFunction(this);

    if (System.isPromise(val)) {
      val = await val;
    }

    this.units.push(val);

    return val;
  } /* addUnit */

  run() {
    let system = this;

    const run = async function() {
      system.timer.response();

      await system.render.start();

      for (let i = 0, count = system.units.length; i < count; i++) {
        system.units[i].response(system);
      }

      system.render.end();

      window.requestAnimationFrame(run);
    };
    window.requestAnimationFrame(run);
  } /* run */
} /* System */
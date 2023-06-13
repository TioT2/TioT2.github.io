import {Material, Texture, UBO, loadShader} from "./material.js";
import {Primitive, Topology, Vertex} from "./primitive.js";

export class Model {
  #gl;
  primitives = [];

  constructor(gl) {
    this.gl = gl;
  } /* constructor */

  async loadGLTF(modelPath) {
    const unpackProperty = function(property, defaultValue = 0) {
      return property === undefined ? defaultValue : property;
    };
    let gl = this.gl;

    try {
      let model = await fetch(modelPath + "/scene.gltf").then(data => data.json());
      // let model = await fetch(modelPath + "/scene.gltf").then(data => data.json());

      // load buffers
      for (let i = 0; i < model.buffers.length; i++) {
        model.buffers[i].data = await fetch(`${modelPath}/${model.buffers[i].uri}`).then(data => data.blob()).then(blob => blob.arrayBuffer());
      }

      // textures (no sampler support)
      for (let i = 0; i < model.textures.length; i++) {
        model.textures[i].texture = new Texture(gl, Texture.UNSIGNED_BYTE, 4);
        model.textures[i].texture.load(`${modelPath}/${model.images[model.textures[i].source].uri}`);
      }

      // dummy materials
      for (let i = 0; i < model.materials.length; i++) {
        model.materials[i].material = new Material(gl, await loadShader(gl, "./bin/shaders/default"));
        model.materials[i].material.textures.push(Texture.defaultChecker(gl));
      }

      // load meshes
      for (let mi = 0; mi < model.meshes.length; mi++) {
        let mesh = model.meshes[mi];

        for (let i = 0; i < mesh.primitives.length; i++) {
          let primitive = mesh.primitives[i];

          let normalAccessor = model.accessors[primitive.attributes.NORMAL];
          let positionAccessor = model.accessors[primitive.attributes.POSITION];
          let texCoordAccessor = model.accessors[primitive.attributes.TEXCOORD_0];

          let resultPrim = new Primitive(gl);

          resultPrim.geometryType = primitive.mode;
          resultPrim.vertexArrayObject = gl.createVertexArray();
          resultPrim.material = model.materials[primitive.material].material;

          gl.bindVertexArray(resultPrim.vertexArrayObject);

          resultPrim.vertexBuffer = gl.createBuffer();
          resultPrim.vertexNumber = positionAccessor.count;

          gl.bindBuffer(gl.ARRAY_BUFFER, resultPrim.vertexBuffer);
          gl.bufferData(gl.ARRAY_BUFFER, model.buffers[model.bufferViews[positionAccessor.bufferView].buffer].data, gl.STATIC_DRAW);

          let size = gl.getBufferParameter(gl.ARRAY_BUFFER, gl.BUFFER_SIZE);

          let shader = resultPrim.material.shader;
          const addArrayPointer = function(name, accessor, shader) {
            let location = gl.getAttribLocation(shader, name);

            if (location != -1) {
              let bufferView = model.bufferViews[accessor.bufferView];
  
              gl.vertexAttribPointer(
                location,
                3,
                accessor.componentType,
                false,
                unpackProperty(bufferView.byteStride, 1),
                unpackProperty(accessor.byteOffset, 0) + unpackProperty(bufferView.byteOffset, 0),
              );
              gl.enableVertexAttribArray(location);
            }
          };
          addArrayPointer("inPosition", positionAccessor, shader);
          addArrayPointer("inTexCoord", texCoordAccessor, shader);
          addArrayPointer("inNormal", normalAccessor, shader);

          // load index buffer
          let indexAccessor = model.accessors[primitive.indices];

          let indexBufferView = model.bufferViews[indexAccessor.bufferView];

          resultPrim.indexNumber = indexAccessor.count;
          resultPrim.indexBuffer = gl.createBuffer();

          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, resultPrim.indexBuffer);
          gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,
            new DataView(
              model.buffers[indexBufferView.buffer].data,
              unpackProperty(indexAccessor.byteOffset, 0) + unpackProperty(indexBufferView.byteOffset),
              unpackProperty(indexBufferView.byteLength)
            ),
            gl.STATIC_DRAW
          );

          this.primitives.push(resultPrim);
        }
      }
    } catch(error) {
      console.error(`Error loading \"${modelPath}\": ${error}`);
    }
  } /* load */

  draw(cameraBuffer = null) {
    for (let i = 0; i < this.primitives.length; i++) {
      primitives[i].draw(cameraBuffer);
    }
  } /* draw */
} /* Model */

/* model.js */
// rollup main.js --file bundle.js --format iife
export default [
  {
    input: './animjs/src/main.js',
    output: {
      file: './animjs/bundle.js',
      format: 'es',
      sourcemap: 'inline',
    }
  },
  {
    input: './sets/main.js',
    output: {
      file: './sets/bundle.js',
      format: 'es',
      sourcemap: 'inline',
    }
  },
]
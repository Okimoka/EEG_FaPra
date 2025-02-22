class ArucoMarker {
    constructor(id, border = 52, scale = 32, gridSize = 8) {
      this.id = id;
      this.border = border;
      this.scale = scale;
      this.gridSize = gridSize;
      this.buffer = null;
  
      this.data = [
      [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,1,1,1,0,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,1,0,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0]
      ][id];
    }
  
    initBuffer() {
      this.buffer = createGraphics(this.scale * this.gridSize, this.scale * this.gridSize);
      this.buffer.noStroke();
      this.buffer.background(255);
      
      for (let i = 0; i < this.gridSize; i++) {
        for (let j = 0; j < this.gridSize; j++) {
          let index = i * this.gridSize + j;
          this.buffer.fill(this.data[index] * 255);
          this.buffer.rect(j * this.scale, i * this.scale, this.scale, this.scale);
        }
      }
    }
  
    draw(width, height) {
      if (!this.buffer) return;
  
      let xpos = [this.border, width - this.scale * 8 - this.border, this.border, width - this.scale * 8 - this.border][this.id];
      let ypos = [this.border, this.border, height - this.scale * 8 - this.border, height - this.scale * 8 - this.border][this.id];
  
      image(this.buffer, xpos, ypos);
    }
  }
  
  class GazePoint {
    constructor(alpha, color = [255, 0, 0]) {
      this.alpha = alpha;
      this.color = color;
      this.prevX = null;
      this.prevY = null;
    }
  
    draw(gazeX, gazeY) {
      if (this.prevX !== null && this.prevY !== null) {
        gazeX = (1 - this.alpha) * this.prevX + this.alpha * gazeX;
        gazeY = (1 - this.alpha) * this.prevY + this.alpha * gazeY;
      }
      renderFractal(gazeX/width, gazeY/height);
      fill(this.color[0], this.color[1], this.color[2]);
      circle(gazeX, gazeY, 10);
      this.prevX = gazeX;
      this.prevY = gazeY;
    }
  }
  
  windowResized = () => {
    resizeCanvas(windowWidth, windowHeight);
  };
  
  
  //Adapted implementation taken from Paul Bourke
  //https://editor.p5js.org/feyzan/sketches/gqukZ3VmF
  
  function renderFractal(mouseX, mouseY) {
      loadPixels();
      
      let ca = (mouseX) * 2 - 1;
      let cb = (mouseY) * 2 - 1;
    
      const w = 4;
      const h = (w * height) / width;
      
      const xmin = -w / 2;
      const ymin = -h / 2;
      
      const dx = w / width;
      const dy = h / height;
      
      const buffer = new Uint8ClampedArray(pixels.length);
      
      let y = ymin;
      let index = 0;
      
      for (let j = 0; j < height; j++) {
        let x = xmin;
        
        for (let i = 0; i < width; i++) {
          let a = x, b = y;
          let n = 0;
          
          while (n < maxIterations) {
            let aa = a * a;
            let bb = b * b;
            
            if (aa + bb > 4) break;
            
            b = 2 * a * b + cb;
            a = aa - bb + ca;
            
            n++;
          }
    
          // Calculate pixel index
          let pix = index * 4;
          if (n === maxIterations) {
            buffer[pix] = buffer[pix + 1] = buffer[pix + 2] = 250;
          } else {
            buffer[pix] = 255 - colorsRed[n];
            buffer[pix + 1] = 255 - colorsGreen[n];
            buffer[pix + 2] = 255 - colorsBlue[n];
          }
          buffer[pix + 3] = 255; // Alpha channel
          
          x += dx;
          index++;
        }
        y += dy;
      }
      
      pixels.set(buffer);
      updatePixels();
  }
  
  
  
  
  setup = () => {
    createCanvas(windowWidth, windowHeight);
    
    pixelDensity(1);
    colorMode(HSB, 1);
  
      // Precompute colors
      for (let n = 0; n < maxIterations; n++) {
        let brightness = sqrt(n / maxIterations);
        let col = color(brightness, 1, brightness);
        
        colorsRed[n] = red(col);
        colorsGreen[n] = green(col);
        colorsBlue[n] = blue(col);
      }
  
    arucoMarkers = [
      new ArucoMarker(0),
      new ArucoMarker(1),
      new ArucoMarker(2),
      new ArucoMarker(3)
    ];
  
    arucoMarkers.forEach(marker => marker.initBuffer());
  };
  
  // gaze smoothing factor 0.2
  let gazePoint = new GazePoint(0.2);
  
  // Colors to be used for each possible iteration count
  const maxIterations = 10;
  const colorsRed = new Uint8Array(maxIterations);
  const colorsGreen = new Uint8Array(maxIterations);
  const colorsBlue = new Uint8Array(maxIterations);
  
  
  draw = () => {
    background(255);
    let gazeX = data?.['x'] * width;
    let gazeY = (1 - data?.['y']) * height;
    gazePoint.draw(gazeX, gazeY);
    arucoMarkers.forEach(marker => marker.draw(width, height));
  };

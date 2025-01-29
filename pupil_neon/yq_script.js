class ArucoMarker {
    constructor(id, border=32, scale=24, gridSize=8) {
  
      this.id = id
      this.border = border
      this.data = [
      [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,1,1,1,0,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,1,0,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0]
      ][id]
  
      this.scale = scale;
      this.gridSize = gridSize;
    }
  
    draw(width, height) {
  
      let xpos = [this.border, width-this.scale*8-this.border, this.border, width-this.scale*8-this.border][this.id]
      let ypos = [this.border, this.border, height-this.scale*8-this.border, height-this.scale*8-this.border][this.id]
  
      noStroke();
      for (let i = 0; i < this.gridSize; i++) {
        for (let j = 0; j < this.gridSize; j++) {
          let index = i * this.gridSize + j;
          fill(this.data[index] * 255);
          rect(xpos + j * this.scale, ypos + i * this.scale, this.scale, this.scale);
        }
      }
    }
  }
  
  
  class GazePoint {
    constructor(alpha, color=[255, 0, 0]) {
      this.alpha = alpha;
      this.color = color;
      this.prevX = null;
      this.prevY = null;
    }
  
    draw(gazeX, gazeY) {
      // Apply smoothing
      if (this.prevX !== null && this.prevY !== null) {
  
        gazeX = (1 - this.alpha) * this.prevX + this.alpha * gazeX;
        gazeY = (1 - this.alpha) * this.prevY + this.alpha * gazeY;
      }
  
      fill(color[0], color[1], color[2]);
      circle(gazeX, gazeY, 10);
      console.log(gazeX, gazeY)
  
      this.prevX = gazeX;
      this.prevY = gazeY;
    }
  }
  
  
  class Present {
  constructor(x, y, w, h, borderColor = 'black') {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.borderColor = borderColor;
    this.isVisible = true;
    this.isEventActive = false; // New attribute to control event image visibility
    this.eventTimeouts = [];  // Array to hold multiple timeouts
    this.eventX = 0; // New: store event image x position
    this.eventY = 0; // New: store event image y position
    this.eventDelays = []; // Array to store delays
  }
  
  draw(gazeX, gazeY) {
    if (!this.isVisible) return;
    image(presentImage, this.x, this.y, this.w, this.h);
    const inside = gazeX > this.x && gazeX < this.x + this.w && gazeY > this.y && gazeY < this.y + this.h;
    stroke(inside ? 'red' : this.borderColor);
    noFill();
    rect(this.x, this.y, this.w, this.h);
  }
  
  triggerEvent() {
    const eventDuration = 2000; // Duration each event is shown
    const numberOfEvents = 5; // Total number of events, adjust as needed
  
    // Calculate onset times
    let cumulativeDelay = 0; // Start with no initial fixed delay
    this.eventDelays = Array.from({length: numberOfEvents}, (_, index) => {
      if (index === 0) {
        cumulativeDelay += random(1000, 3000); // Initial event after a random delay
      } else {
        cumulativeDelay += eventDuration + random(1000, 3000); // Subsequent events
      }
      return cumulativeDelay;
    });
  
    // Send all onset times to the server in one go
    communicateToServer(this.eventDelays);
  
    // Schedule each event
    this.eventDelays.forEach((delay, index) => {
      this.eventTimeouts.push(setTimeout(() => {
        this.showEvent();
        setTimeout(() => this.hideEvent(), eventDuration);
      }, delay));
    });
  }
  
  showEvent() {
    this.isEventActive = true;
    this.eventX = random(0, width - 100); // Adjust for your eventImage width
    this.eventY = random(0, height - 100); // Adjust for your eventImage height
  }
  
  hideEvent() {
    this.isEventActive = false;
  }
  
    // To clear all timeouts if needed
  clearEventTimeouts() {
    this.eventTimeouts.forEach(clearTimeout);
    this.eventTimeouts = [];
  }
  
  }
  
  
  function communicateToServer(onsetTimes) {
  
    const data = {
      event: "multipleEvents",
      onsets: onsetTimes,
      timestamp: Date.now()
    };
  
    // Send a POST request to the Python server
    httpPost('http://localhost:5000/event', 'json', data, function(result) {
        console.log('Server response:', result);
    }, function(error) {
        console.error('Error:', error);
    });
  }
  
  inside_global = false
  
  document.addEventListener('keydown', function(event) {
      if (event.key === 'h' || event.key === 'H') {
        present=presents[1]
          present.isVisible = false;
        present.triggerEvent();
      }
  });
  
  function blinked(gazeX, gazeY) {
  
  /*
  presents.forEach((present) => {
    let inside = inside_global//gazeX > present.x && gazeX < present.x + present.w && gazeY > present.y && gazeY < present.y + present.h;
  
    if (inside && present.isVisible) {
        console.log('Blink detected while looking at a present');
        present.isVisible = false;
        present.triggerEvent();
    }
    });
    previousBlinkCount = data?.['blink_count']; // Update the previous blink count
  */
  }
  
  function drawEventImage(x, y) {
    if (present.isEventActive) {
        image(eventImage, x, y, 10, 10); // Adjust size if needed
    }
  }
  
  
  windowResized = () => {
    resizeCanvas(windowWidth, windowHeight);
  }
  
  setup = () => {
    createCanvas(windowWidth, windowHeight);
    //data['blink_count'] = 0
  
  
  setTimeout(() => {
    //blinked()
  }, 10000);
  }
   
  let presentImage; 
  let eventImage;
  
  preload = () => {
    presentImage = loadImage('http://127.0.0.1:5000/images/images/Present.png'); // Make sure the path is correct
    eventImage = loadImage('http://127.0.0.1:5000/images/images/Event.png'); // Make sure the path is correct
  }
  
  let arucoMarkers = [
    new ArucoMarker(0),
    new ArucoMarker(1),
    new ArucoMarker(2),
    new ArucoMarker(3)
  ];
  
  
  let presents = [
    new Present(8000, 500, 200, 100),
    new Present(3000, 100, 150, 150)
  ];
  
  let gazePoint = new GazePoint(0.2);
  let previousBlinkCount = 0;
  
  // Code that is constantly being updated
  draw = () => {
      background(255,255,255)
  
      let gazeX = data?.['X'] * width;
      let gazeY = (1-data?.['Y']) * height;
  
      arucoMarkers.forEach(marker => marker.draw(width, height))
      presents.forEach(present => present.draw(gazeX, gazeY));
      gazePoint.draw(gazeX, gazeY)
  
      // Check for blink
      if (data?.['blink_count'] > previousBlinkCount) {
        blinked(gazeX, gazeY)
      }
  
      arucoMarkers.forEach(marker => marker.draw(width, height));
  
      
      presents.forEach(present => {
        present.draw(gazeX, gazeY);
        if (present.isEventActive) {
            image(eventImage, present.eventX, present.eventY); // Draw event image at stored coordinates
        }
    });
  
  
  
      
  }
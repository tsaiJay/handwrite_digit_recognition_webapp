const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const colorPicker = document.getElementById('colorPicker');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const strokeWidth = document.getElementById('strokeWidth');
const strokeWidthValue = document.getElementById('strokeWidthValue');

let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Set initial canvas background to white
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Drawing functions
function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function draw(e) {
    if (!isDrawing) return;
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.strokeStyle = colorPicker.value;
    ctx.lineWidth = strokeWidth.value;
    ctx.lineCap = 'round';
    ctx.stroke();
    
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function stopDrawing() {
    isDrawing = false;
}

// Event listeners for drawing
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Clear canvas
clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});

// Send drawing
sendBtn.addEventListener('click', () => {
    const imageData = canvas.toDataURL('image/png');
    // Update prediction text to loading state
    const predictionResult = document.getElementById('predictionResult');
    predictionResult.textContent = 'Processing...';
    
    // Send to Python server
    fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: imageData
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            predictionResult.textContent = data.prediction;
        } else {
            predictionResult.textContent = 'Error: 111' + data.error;
        }
    })
    .catch(error => {
        predictionResult.textContent = 'Error: server is not running. ' + error.message;
        console.error('Error: server is not running. ', error);
    });
});

// Add this event listener
strokeWidth.addEventListener('input', function() {
    strokeWidthValue.textContent = this.value;
}); 
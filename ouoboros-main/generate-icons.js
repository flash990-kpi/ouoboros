// Generate PWA icons with 5-pixel square pattern
const fs = require('fs');
const { createCanvas } = require('canvas');

function generateIcon(size) {
    const canvas = createCanvas(size, size);
    const ctx = canvas.getContext('2d');
    
    // Background
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, size, size);
    
    // Calculate pixel size for 5-pixel pattern
    const pixelSize = size / 8;
    const offset = size / 8;
    
    // Draw 5 pixels in square pattern (like a cross)
    ctx.fillStyle = '#00ffcc';
    
    // Center pixel
    ctx.fillRect(offset * 3.5, offset * 3.5, pixelSize, pixelSize);
    
    // Top
    ctx.fillRect(offset * 3.5, offset * 1.5, pixelSize, pixelSize);
    
    // Bottom
    ctx.fillRect(offset * 3.5, offset * 5.5, pixelSize, pixelSize);
    
    // Left
    ctx.fillRect(offset * 1.5, offset * 3.5, pixelSize, pixelSize);
    
    // Right
    ctx.fillRect(offset * 5.5, offset * 3.5, pixelSize, pixelSize);
    
    return canvas;
}

// Generate icons
const icon192 = generateIcon(192);
const icon512 = generateIcon(512);

icon192.toBuffer((err, buf) => {
    if (err) throw err;
    fs.writeFileSync('icon-192.png', buf);
    console.log('Generated icon-192.png');
});

icon512.toBuffer((err, buf) => {
    if (err) throw err;
    fs.writeFileSync('icon-512.png', buf);
    console.log('Generated icon-512.png');
});

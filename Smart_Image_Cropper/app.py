# app.py
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw
import numpy as np
import base64
import io
import os
import re
from datetime import datetime
import cv2
import random

app = Flask(__name__)

SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'webp', 'gif']

def process_image(img, mask):
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Create transparent background
    transparent = Image.new('RGBA', img.size, (0, 0, 0, 0))
    
    # Apply mask to create transparency
    img.putalpha(mask)
    
    # Composite the image
    result = Image.alpha_composite(transparent, img)
    
    # Find bounding box of non-transparent pixels
    bbox = result.getbbox()
    if bbox:
        result = result.crop(bbox)
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop', methods=['POST'])
def crop_image():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        shape_type = data['shapeType']
        points = data['points']
        
        # Get original format
        format_match = re.match(r'data:image/(\w+);base64', data['image'])
        image_format = format_match.group(1) if format_match else 'png'
        
        if image_format.lower() not in SUPPORTED_FORMATS:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported image format: {image_format}'
            }), 400
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        
        # Create mask
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        
        if shape_type == 'circle':
            center = points[0]  # First point is center
            radius = points[1]  # Second point determines radius
            draw.ellipse([
                center['x'] - radius,
                center['y'] - radius,
                center['x'] + radius,
                center['y'] + radius
            ], fill=255)
        elif shape_type == 'polygon':
            # Convert points to flat tuple for PIL
            polygon_points = [(p['x'], p['y']) for p in points]
            draw.polygon(polygon_points, fill=255)
        
        # Process and crop image
        result = process_image(img, mask)
        
        # Save to buffer
        buffer = io.BytesIO()
        result.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'status': 'success',
            'image': f'data:image/png;base64,{img_str}'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/save', methods=['POST'])
def save_image():
    data = request.json
    image_data = data['image'].split(',')[1]
    
    # Detect original format
    format_match = re.match(r'data:image/(\w+);base64', data['image'])
    image_format = format_match.group(1) if format_match else 'png'
    
    # Convert base64 to image
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes))
    
    # Save with original format if possible, fallback to PNG for transparency
    save_format = image_format if image_format.lower() in ['png', 'webp'] else 'png'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cropped_image_{timestamp}.{save_format}"
    save_path = os.path.join('static', 'saved_images', filename)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path, format=save_format, quality=100)
    
    return jsonify({
        'status': 'success',
        'filename': filename,
        'path': save_path
    })

def apply_glitch_effect(img):
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    
    # Create glitch effect
    channels = list(cv2.split(img_cv))  # Convert tuple to list for modification
    
    # Random offset for channels
    for i in range(3):  # Only affect RGB channels, not alpha
        offset = random.randint(-20, 20)
        if offset > 0:
            channels[i] = np.roll(channels[i], offset, axis=1)
    
    # Add color shift
    for i in range(3):
        if random.random() > 0.5:
            channels[i] = np.roll(channels[i], random.randint(-15, 15), axis=0)
    
    # Merge channels back
    glitched = cv2.merge(channels)
    
    # Add some noise and compression artifacts
    noise = np.random.normal(0, 15, glitched.shape[:2]).astype(np.uint8)
    noise_rgba = cv2.merge([noise, noise, noise, np.zeros_like(noise)])
    glitched = cv2.add(glitched, noise_rgba)
    
    # Add random blocks of shifted pixels
    for _ in range(random.randint(2, 5)):
        x1 = random.randint(0, glitched.shape[1] - 50)
        y1 = random.randint(0, glitched.shape[0] - 50)
        width = random.randint(10, 50)
        height = random.randint(5, 20)
        shift = random.randint(-20, 20)
        
        block = glitched[y1:y1+height, x1:x1+width].copy()
        glitched[y1:y1+height, x1:x1+width] = np.roll(block, shift, axis=1)
    
    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(glitched, cv2.COLOR_BGRA2RGBA))

def apply_oil_painting(img):
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
    
    # Parameters for oil painting effect
    kernel_size = 7
    dynRatio = 1
    
    # Apply bilateral filter for smoothing while preserving edges
    smooth = cv2.bilateralFilter(img_cv, 9, 75, 75)
    
    # Create oil painting effect using median blur and edge preservation
    oil = cv2.medianBlur(smooth, kernel_size)
    
    # Enhance edges
    edges = cv2.Canny(cv2.cvtColor(oil, cv2.COLOR_BGR2GRAY), 100, 200)
    edges = cv2.dilate(edges, None)
    
    # Combine edge information with the oil effect
    oil_with_edges = oil.copy()
    oil_with_edges[edges > 0] = oil_with_edges[edges > 0] * 0.7
    
    # Enhance color saturation
    hsv = cv2.cvtColor(oil_with_edges, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.2  # Increase saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    oil_with_edges = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Add alpha channel
    alpha = np.full((oil_with_edges.shape[0], oil_with_edges.shape[1]), 255, dtype=np.uint8)
    oil_rgba = cv2.cvtColor(oil_with_edges, cv2.COLOR_BGR2BGRA)
    oil_rgba[:, :, 3] = alpha
    
    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(oil_rgba, cv2.COLOR_BGRA2RGBA))

def apply_neon_effect(img):
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    
    # Detect edges
    edges = cv2.Canny(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2GRAY), 100, 200)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    
    # Create neon effect
    neon = img_cv.copy()
    neon[edges > 0] = [0, 255, 255, 255]  # Yellow neon
    neon = cv2.GaussianBlur(neon, (3, 3), 0)
    
    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(neon, cv2.COLOR_BGRA2RGBA))

def apply_ascii_effect(img):
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2GRAY)
    
    # ASCII characters from dark to light
    ascii_chars = "@%#*+=-:. "
    
    # Resize image to maintain aspect ratio
    height, width = img_cv.shape
    new_width = 100
    new_height = int((height * new_width) / width)
    img_cv = cv2.resize(img_cv, (new_width, new_height))
    
    # Create ASCII art
    ascii_img = np.zeros((new_height * 10, new_width * 6, 3), dtype=np.uint8)
    ascii_img.fill(255)
    
    for i in range(new_height):
        for j in range(new_width):
            pixel_value = img_cv[i, j]
            char_idx = int(pixel_value / 255 * (len(ascii_chars) - 1))
            cv2.putText(ascii_img, ascii_chars[char_idx], (j * 6, i * 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    # Convert back to PIL
    return Image.fromarray(ascii_img)

def apply_pixel_effect(img):
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    
    # Define pixel size
    pixel_size = 10
    
    # Resize down and up to create pixelation
    height, width = img_cv.shape[:2]
    small = cv2.resize(img_cv, (width // pixel_size, height // pixel_size),
                      interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(pixelated, cv2.COLOR_BGRA2RGBA))

# def apply_double_exposure(img):
#     # Convert PIL to OpenCV format (handling RGBA properly)
#     img_cv = np.array(img)
    
#     # Ensure the image is in the correct format (BGRA or BGR)
#     if img_cv.shape[-1] == 4:  # If it has an alpha channel
#         bgr = img_cv[:, :, :3]  # Extract RGB channels only
#         alpha_channel = img_cv[:, :, 3]  # Store alpha separately
#     else:
#         bgr = img_cv
#         alpha_channel = None

#     # Create an inverted copy (only for the BGR part)
#     inverted = cv2.bitwise_not(bgr)

#     # Blend images
#     alpha = 0.5
#     double_exposed = cv2.addWeighted(bgr, alpha, inverted, 1 - alpha, 0)

#     # Add some brightness and contrast
#     double_exposed = cv2.convertScaleAbs(double_exposed, alpha=1.2, beta=10)

#     # Reattach the alpha channel if needed
#     if alpha_channel is not None:
#         double_exposed = cv2.merge((double_exposed, alpha_channel))

#     # Convert back to PIL
#     mode = "RGBA" if alpha_channel is not None else "RGB"
#     return Image.fromarray(cv2.cvtColor(double_exposed, cv2.COLOR_BGR2RGB), mode)

@app.route('/apply_effect', methods=['POST'])
def apply_effect():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        effect_type = data['effectType']
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        
        # Apply selected effect
        effect_functions = {
            'glitch': apply_glitch_effect,
            'neon': apply_neon_effect,
            'ascii': apply_ascii_effect,
            'pixel': apply_pixel_effect,
            'oil': apply_oil_painting,
            'retrowave': apply_retrowave_effect,
            'dream': apply_dream_effect,
            'comic': apply_comic_effect,
            'watercolor': apply_watercolor_effect
        }
        
        if effect_type in effect_functions:
            result = effect_functions[effect_type](img)
        else:
            return jsonify({'status': 'error', 'message': 'Invalid effect type'})
        
        # Save to buffer
        buffer = io.BytesIO()
        result.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'status': 'success',
            'image': f'data:image/png;base64,{img_str}'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

# Add these new effect functions to your app.py

def apply_retrowave_effect(img):
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    
    # Split into channels
    b, g, r, a = cv2.split(img_cv)
    
    # Enhance blue and red channels for retrowave look
    b = cv2.addWeighted(b, 1.5, np.zeros_like(b), 0, 0)
    r = cv2.addWeighted(r, 1.3, np.zeros_like(r), 0, 30)
    
    # Add purple tint
    purple_tint = np.full_like(b, 20)
    b = cv2.add(b, purple_tint)
    r = cv2.add(r, purple_tint)
    
    # Merge channels back
    retrowave = cv2.merge([b, g, r, a])
    
    # Add gradient
    height, width = retrowave.shape[:2]
    gradient = np.linspace(0, 1, height)[:, np.newaxis]
    gradient = np.tile(gradient, (1, width))
    gradient = (gradient * 50).astype(np.uint8)
    
    # Apply gradient to image
    retrowave[:,:,2] = cv2.add(retrowave[:,:,2], gradient)
    
    # Add scan lines
    scan_lines = np.zeros((height, width), dtype=np.uint8)
    scan_lines[::3, :] = 20
    retrowave = cv2.subtract(retrowave, cv2.merge([scan_lines, scan_lines, scan_lines, np.zeros_like(scan_lines)]))
    
    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(retrowave, cv2.COLOR_BGRA2RGBA))

def apply_dream_effect(img):
    # Convert PIL to OpenCV format and ensure BGR
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
    
    # Create dreamy blur effect
    blur = cv2.GaussianBlur(img_cv, (21, 21), 0)
    
    # Enhance colors
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.4, 0, 255)  # Increase saturation
    hsv = hsv.astype(np.uint8)
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Add soft glow
    glow = cv2.addWeighted(enhanced, 0.7, blur, 0.3, 0)
    
    # Create light leak effect
    height, width = glow.shape[:2]
    light_leak = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient light leak
    for i in range(3):  # BGR channels
        light_leak[:,:,i] = np.random.normal(128, 50, (height, width)).clip(0, 255).astype(np.uint8)
    
    light_leak = cv2.GaussianBlur(light_leak, (99, 99), 30)
    
    # Blend light leak with image
    dream = cv2.addWeighted(glow, 0.8, light_leak, 0.2, 0)
    
    # Convert back to RGBA
    dream_rgba = cv2.cvtColor(dream, cv2.COLOR_BGR2BGRA)
    dream_rgba[:,:,3] = 255  # Set full opacity
    
    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(dream_rgba, cv2.COLOR_BGRA2RGBA))

def apply_comic_effect(img):
    # Convert PIL to OpenCV format and ensure BGR
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
    
    # Create edge mask using Canny instead of adaptive threshold
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.dilate(edges, None)
    
    # Reduce colors using pyrDown/pyrUp
    color = cv2.pyrDown(img_cv)
    color = cv2.pyrUp(color)
    
    # Apply median blur for cartoon effect
    color = cv2.medianBlur(color, 7)
    
    # Quantize colors
    h, w = img_cv.shape[:2]
    color = color.reshape((-1, 3))
    color = np.float32(color)
    
    # Reduce color palette
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    K = 8
    _, labels, centers = cv2.kmeans(color, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    color = centers[labels.flatten()]
    color = color.reshape((h, w, 3))
    
    # Combine edges with color
    edges_inv = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2BGR)
    comic = cv2.bitwise_and(color, edges_inv)
    
    # Convert to RGBA
    comic_rgba = cv2.cvtColor(comic, cv2.COLOR_BGR2BGRA)
    comic_rgba[:,:,3] = 255  # Set full opacity
    
    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(comic_rgba, cv2.COLOR_BGRA2RGBA))

def apply_watercolor_effect(img):
    # Convert PIL to OpenCV format and ensure BGR
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
    
    # Apply median blur for initial smoothing
    smoothed = cv2.medianBlur(img_cv, 7)
    
    # Apply bilateral filter with safe parameters
    bilateral1 = cv2.bilateralFilter(smoothed, 5, 50, 50)
    bilateral2 = cv2.bilateralFilter(bilateral1, 5, 50, 50)
    
    # Create edge mask
    gray = cv2.cvtColor(bilateral2, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 9, 7)
    edges = 255 - edges
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Reduce color palette
    h, w = img_cv.shape[:2]
    small = cv2.resize(bilateral2, (w//2, h//2))
    color = small.reshape((-1, 3))
    color = np.float32(color)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    K = 9
    _, labels, centers = cv2.kmeans(color, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    color = centers[labels.flatten()]
    quantized = color.reshape((h//2, w//2, 3))
    quantized = cv2.resize(quantized, (w, h))
    
    # Blend quantized image with edges
    watercolor = cv2.addWeighted(quantized, 0.7, edges, 0.3, 0)
    
    # Add subtle texture
    texture = np.random.normal(0, 2, watercolor.shape).astype(np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    watercolor = cv2.add(watercolor, texture)
    
    # Convert to RGBA
    watercolor_rgba = cv2.cvtColor(watercolor, cv2.COLOR_BGR2BGRA)
    watercolor_rgba[:,:,3] = 255  # Set full opacity
    
    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(watercolor_rgba, cv2.COLOR_BGRA2RGBA))

if __name__ == '__main__':
    app.run(debug=True)
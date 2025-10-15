from flask import Flask, request, jsonify
import requests
import base64
import io
import os
from PIL import Image
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# API Keys (load from environment variables)
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY', '')

# API Endpoints
ROBOFLOW_BASE_URL = "https://detect.roboflow.com"

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

def process_uploaded_image(file_or_base64):
    """Process uploaded image from file or base64"""
    if isinstance(file_or_base64, str):
        return base64_to_image(file_or_base64)
    else:
        return Image.open(file_or_base64.stream)

@app.route('/')
def home():
    """API documentation endpoint"""
    return jsonify({
        "message": "Computer Vision API - Roboflow Integration",
        "version": "2.0.0",
        "description": "Object detection API powered by Roboflow",
        "supported_providers": {
            "roboflow": {
                "enabled": bool(ROBOFLOW_API_KEY),
                "features": ["object_detection", "classification", "segmentation"]
            }
        },
        "endpoints": {
            "/": "API documentation",
            "/health": "Health check",
            "/roboflow/detect": "Roboflow object detection"
        },
        "supported_formats": ["PNG", "JPG", "JPEG", "GIF", "BMP", "TIFF", "WEBP"],
        "max_file_size": "16MB"
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "api_keys_configured": {
            "roboflow": bool(ROBOFLOW_API_KEY)
        }
    })

@app.route('/roboflow/detect', methods=['POST'])
def roboflow_detect():
    """Object detection using Roboflow"""
    try:
        if not ROBOFLOW_API_KEY:
            return jsonify({"error": "Roboflow API key not configured"}), 400
        
        # Get parameters
        model_id = request.form.get('model_id') if 'file' in request.files else request.json.get('model_id', 'coco/3')
        confidence = request.form.get('confidence', 40) if 'file' in request.files else request.json.get('confidence', 40)
        
        # Get image
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '' or not allowed_file(file.filename):
                return jsonify({"error": "Invalid file"}), 400
            image = process_uploaded_image(file)
        elif 'image' in request.json:
            image = process_uploaded_image(request.json['image'])
        else:
            return jsonify({"error": "No image provided"}), 400
        
        # Convert to base64
        img_base64 = image_to_base64(image)
        
        # Make request to Roboflow
        # Split model_id into workspace/version format
        model_parts = model_id.split('/')
        if len(model_parts) == 2:
            workspace, version = model_parts
            url = f"https://detect.roboflow.com/{workspace}/{version}?api_key={ROBOFLOW_API_KEY}&confidence={confidence}"
        else:
            url = f"{ROBOFLOW_BASE_URL}/{model_id}?api_key={ROBOFLOW_API_KEY}&confidence={confidence}"
        
        response = requests.post(
            url,
            data=img_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                "provider": "roboflow",
                "model": model_id,
                "predictions": result.get('predictions', []),
                "count": len(result.get('predictions', []))
            })
        else:
            return jsonify({"error": "Roboflow API error", "details": response.text}), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Computer Vision API - Roboflow Integration")
    print("=" * 60)
    print("\nConfigured API Keys:")
    print(f"  Roboflow: {'✓' if ROBOFLOW_API_KEY else '✗'}")
    print("\nStarting Flask server...")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

#!/usr/bin/env python3
"""
Test Roboflow API with a real image
Place any image file in the same directory and name it 'test.jpg'
Or this script will download a sample image
"""

import requests
import os

# Download a sample image if test.jpg doesn't exist
if not os.path.exists('test.jpg'):
    print("Downloading sample image...")
    img_url = "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=800"
    response = requests.get(img_url)
    with open('test.jpg', 'wb') as f:
        f.write(response.content)
    print("✓ Sample image downloaded: test.jpg")
else:
    print("✓ Using existing test.jpg")

# Test Roboflow endpoint
print("\n" + "="*60)
print("Testing Roboflow Object Detection with Real Image")
print("="*60)

try:
    with open('test.jpg', 'rb') as f:
        files = {'file': f}
        data = {
            'model_id': 'coco/3',
            'confidence': 40
        }
        
        print("\nSending request...")
        response = requests.post(
            'http://localhost:5000/roboflow/detect',
            files=files,
            data=data,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ SUCCESS!")
            print(f"\nProvider: {result.get('provider')}")
            print(f"Model: {result.get('model')}")
            print(f"Objects detected: {result.get('count', 0)}")
            
            if result.get('predictions'):
                print("\n🎯 Detected objects:")
                for i, pred in enumerate(result['predictions'], 1):
                    class_name = pred.get('class', 'Unknown')
                    confidence = pred.get('confidence', 0)
                    bbox = pred.get('bbox', {})
                    print(f"\n  {i}. {class_name}")
                    print(f"     Confidence: {confidence:.1%}")
                    if bbox:
                        print(f"     Location: x={bbox.get('x', 0):.0f}, y={bbox.get('y', 0):.0f}")
            else:
                print("\nNo objects detected in the image.")
        else:
            print("\n✗ ERROR!")
            print(f"Response: {response.json()}")
            
except Exception as e:
    print(f"\n✗ ERROR: {e}")

print("\n" + "="*60)
print("Test complete!")
print("="*60)
print("\n💡 Tip: Try with your own images!")
print("   Just replace 'test.jpg' with any photo.")

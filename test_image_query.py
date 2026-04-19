"""Quick test: image-only query to /api/search"""
import os
import requests

img_dir = r"C:\Users\Kartikeya Singh\OneDrive - LNMIIT\Desktop\archive\DeepFashion2\deepfashion2_original_images\test\test\image"
files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")][:1]
print(f"Using: {files[0]}")
path = os.path.join(img_dir, files[0])

with open(path, "rb") as f:
    r = requests.post(
        "http://127.0.0.1:5000/api/search",
        files={"image": ("test.jpg", f, "image/jpeg")},
        data={"text_weight": "0.5"},
    )

print(f"Status: {r.status_code}")
data = r.json()
print(f"Mode: {data.get('mode', 'N/A')}")
print(f"Error: {data.get('error', 'None')}")
print(f"Results: {len(data.get('results', []))}")
for i, x in enumerate(data.get("results", [])):
    print(f"  {i+1}. {x['filename']} score={x['final_score']} src={x['source']}")

# import base64
# import io
# import numpy as np
# import cv2
# import torch
# from flask import Flask, request, jsonify
# from facenet_pytorch import InceptionResnetV1, MTCNN
# from PIL import Image

# app = Flask(__name__)

# # Initialize MTCNN and FaceNet
# mtcnn = MTCNN(keep_all=True)
# resnet = InceptionResnetV1(pretrained="vggface2").eval()

# # Example known faces
# known_faces = {
#     "Yash": "img/yash.jpg",
#     # Add more known faces here
# }

# # Encode known faces
# known_encodings = []
# known_names = []

# def detect_and_encode(img):
#     # Detect face
#     boxes, _ = mtcnn.detect(img)
#     if boxes is not None:
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box)
#             face = img[y1:y2, x1:x2]
#             if face.size == 0:
#                 continue
#             # Resize to 160x160 for FaceNet
#             face = cv2.resize(face, (160, 160))
#             # Normalize
#             face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
#             face_tensor = torch.tensor(face).unsqueeze(0)
#             # Get encoding
#             encoding = resnet(face_tensor).detach().numpy().flatten()
#             return encoding
#     return None

# # Pre-calculate known face encodings
# for name, path in known_faces.items():
#     img_bgr = cv2.imread(path)
#     if img_bgr is not None:
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         enc = detect_and_encode(img_rgb)
#         if enc is not None:
#             known_encodings.append(enc)
#             known_names.append(name)

# @app.route("/face_login", methods=["POST"])
# def face_login():
#     data = request.json
#     image_data = data.get("image", "")
#     if not image_data:
#         return jsonify({"success": False, "message": "No image data"}), 400

#     # Decode base64 image
#     try:
#         header, encoded = image_data.split(",", 1)
#         img_bytes = base64.b64decode(encoded)
#         pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#         img_np = np.array(pil_img)
#     except Exception as e:
#         return jsonify({"success": False, "message": "Invalid image"}), 400

#     # Encode test face
#     test_enc = detect_and_encode(img_np)
#     if test_enc is None:
#         return jsonify({"success": False, "message": "No face detected"}), 400

#     # Compare with known encodings
#     threshold = 0.6
#     distances = np.linalg.norm(np.array(known_encodings) - test_enc, axis=1)
#     min_idx = np.argmin(distances)
#     if distances[min_idx] < threshold:
#         return jsonify({"success": True, "user": known_names[min_idx]})
#     else:
#         return jsonify({"success": False, "message": "Face not recognized"}), 401

# if __name__ == "__main__":
#     app.run(debug=True)
import base64
import io
import numpy as np
import cv2
import torch
from flask import Flask, request, jsonify
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

app = Flask(__name__)

# Initialize MTCNN and FaceNet
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained="vggface2").eval()

# Example known faces dictionary
known_faces = {
    "Yash": "img/yash.jpg",
    # Add more known faces here
}

# Lists to store known encodings and their corresponding names
known_encodings = []
known_names = []

def detect_and_encode(img):
    """
    Detects a face in the image using MTCNN, crops it, resizes it,
    and then computes its encoding using FaceNet.
    """
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            # Resize to 160x160 as required by FaceNet
            face = cv2.resize(face, (160, 160))
            # Normalize and rearrange dimensions from HxWxC to CxHxW
            face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
            face_tensor = torch.tensor(face).unsqueeze(0)
            # Use no_grad for inference to improve performance
            with torch.no_grad():
                encoding = resnet(face_tensor).detach().numpy().flatten()
            return encoding
    return None

# Pre-calculate known face encodings
for name, path in known_faces.items():
    img_bgr = cv2.imread(path)
    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        enc = detect_and_encode(img_rgb)
        if enc is not None:
            known_encodings.append(enc)
            known_names.append(name)
        else:
            print(f"Warning: No face detected in {path}.")
    else:
        print(f"Warning: Could not load image from {path}.")

@app.route("/face_login", methods=["POST"])
def face_login():
    data = request.json
    image_data = data.get("image", "")
    if not image_data:
        return jsonify({"success": False, "message": "No image data"}), 400

    # Decode the base64 image data
    try:
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(pil_img)
    except Exception as e:
        return jsonify({"success": False, "message": "Invalid image"}), 400

    # Encode the detected face in the provided image
    test_enc = detect_and_encode(img_np)
    if test_enc is None:
        return jsonify({"success": False, "message": "No face detected"}), 400

    # Compare test encoding with known face encodings using Euclidean distance
    threshold = 0.6
    distances = np.linalg.norm(np.array(known_encodings) - test_enc, axis=1)
    min_idx = np.argmin(distances)
    if distances[min_idx] < threshold:
        return jsonify({"success": True, "user": known_names[min_idx]})
    else:
        return jsonify({"success": False, "message": "Face not recognized"}), 401

if __name__ == "__main__":
    app.run(debug=True)
python -m venv env

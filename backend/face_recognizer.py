import cv2
import numpy as np
import pickle
import base64
import os
from io import BytesIO
from PIL import Image

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

FACE_WIDTH, FACE_HEIGHT = 150, 150


def process_base64_image(image_data):
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    img_bytes = base64.b64decode(image_data)
    img = Image.open(BytesIO(img_bytes))
    img = np.array(img)
    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def detect_face_roi(image_data):
    img = process_base64_image(image_data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        alt_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
        faces = alt_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
    if len(faces) == 0:
        return None, None, "No face detected in the image"
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    (x, y, w, h) = faces[0]
    x = max(0, x - int(w * 0.1))
    y = max(0, y - int(h * 0.1))
    w = min(gray.shape[1] - x, int(w * 1.2))
    h = min(gray.shape[0] - y, int(h * 1.2))
    face_roi = gray[y:y + h, x:x + w]
    face_roi = cv2.resize(face_roi, (FACE_WIDTH, FACE_HEIGHT))
    face_roi = cv2.equalizeHist(face_roi)
    return face_roi, img, "Success"


def extract_face_encoding(image_data):
    try:
        if FACE_RECOGNITION_AVAILABLE:
            img = process_base64_image(image_data)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img, model='hog')
            if not face_locations:
                return None, "No face detected in the image"
            encodings = face_recognition.face_encodings(rgb_img, face_locations)
            if not encodings:
                return None, "Could not extract face features"
            encoding_bytes = pickle.dumps(encodings[0])
            encoding_b64 = base64.b64encode(encoding_bytes).decode('utf-8')
            return encoding_b64, "Success"
        else:
            face_roi, _, msg = detect_face_roi(image_data)
            if face_roi is None:
                return None, msg
            hog_features = extract_hog_features(face_roi)
            encoding_bytes = pickle.dumps(hog_features)
            encoding_b64 = base64.b64encode(encoding_bytes).decode('utf-8')
            return encoding_b64, "Success"
    except Exception as e:
        return None, f"Error processing image: {str(e)}"


def extract_hog_features(face_roi):
    hog = cv2.HOGDescriptor()
    features = hog.compute(face_roi)
    features = features.flatten()
    features = features / (np.linalg.norm(features) + 1e-10)
    return features


def compare_faces(known_encoding_b64, unknown_encoding_b64, tolerance=0.6):
    try:
        known_bytes = base64.b64decode(known_encoding_b64)
        unknown_bytes = base64.b64decode(unknown_encoding_b64)
        known_data = pickle.loads(known_bytes)
        unknown_data = pickle.loads(unknown_bytes)

        if FACE_RECOGNITION_AVAILABLE:
            results = face_recognition.compare_faces(
                [known_data], unknown_data, tolerance=tolerance
            )
            distance = face_recognition.face_distance(
                [known_data], unknown_data
            )[0]
            return bool(results[0]), float(distance)
        else:
            distance = np.linalg.norm(known_data - unknown_data)
            max_dist = np.sqrt(len(known_data))
            norm_distance = float(distance / max_dist)
            threshold = 1.0 - tolerance
            return norm_distance < threshold, norm_distance
    except Exception as e:
        return False, 1.0


def find_best_match(unknown_encoding_b64, employees_list, tolerance=0.6):
    from backend.ml_trainer import predict_with_models, decode_encoding

    ml_predictions = predict_with_models(unknown_encoding_b64)

    best_match = None
    best_distance = 1.0

    if FACE_RECOGNITION_AVAILABLE:
        for emp in employees_list:
            if not emp.get('face_encoding'):
                continue
            is_match, distance = compare_faces(
                emp['face_encoding'], unknown_encoding_b64, tolerance
            )
            if is_match and distance < best_distance:
                best_distance = distance
                best_match = emp
    else:
        try:
            unknown_bytes = base64.b64decode(unknown_encoding_b64)
            unknown_face = pickle.loads(unknown_bytes)

            images = []
            labels = []
            label_map = {}

            for i, emp in enumerate(employees_list):
                if not emp.get('face_encoding'):
                    continue
                known_bytes = base64.b64decode(emp['face_encoding'])
                known_face = pickle.loads(known_bytes)
                images.append(known_face)
                labels.append(i)
                label_map[i] = emp

            if len(images) < 1:
                return best_match, best_distance

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(images, np.array(labels, dtype=np.int32))

            label_id, confidence = recognizer.predict(unknown_face)

            threshold_map = 100 - (tolerance * 100)
            if confidence < threshold_map and label_id in label_map:
                best_match = label_map[label_id]
                best_distance = confidence / 100.0
        except Exception:
            pass

    if best_match is None and ml_predictions:
        from sklearn.preprocessing import LabelEncoder
        import joblib
        from backend.ml_trainer import LABEL_ENCODER_PATH

        for model_name, pred in ml_predictions.items():
            if 'predicted_id' in pred and pred['confidence'] > 60:
                predicted_id = pred['predicted_id']
                candidate = next((e for e in employees_list if str(e['_id']) == predicted_id), None)
                if candidate and (best_match is None or pred['confidence'] > (1 - best_distance) * 100):
                    if best_match is None or pred['confidence'] > 70:
                        best_match = candidate
                        best_distance = 1.0 - (pred['confidence'] / 100.0)
                        break

    return best_match, best_distance


def save_face_image(image_data, employee_id):
    try:
        face_roi, full_img, msg = detect_face_roi(image_data)
        if face_roi is None:
            return None
        upload_dir = os.path.join('uploads', 'faces')
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, f'employee_{employee_id}.jpg')
        cv2.imwrite(filepath, face_roi)
        return filepath
    except Exception as e:
        return None

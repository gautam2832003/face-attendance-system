import os
import pickle
import base64
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timezone

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

RF_MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
SVM_MODEL_PATH = os.path.join(MODELS_DIR, 'svm_model.pkl')
DT_MODEL_PATH = os.path.join(MODELS_DIR, 'decision_tree_model.pkl')
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
TRAINING_META_PATH = os.path.join(MODELS_DIR, 'training_meta.pkl')


def decode_encoding(encoding_b64):
    try:
        encoding_bytes = base64.b64decode(encoding_b64)
        return pickle.loads(encoding_bytes)
    except Exception as e:
        return None


def generate_csv_dataset(employees_list):
    if not employees_list:
        return None
    rows = []
    for emp in employees_list:
        if not emp.get('face_encoding'):
            continue
        encoding = decode_encoding(emp['face_encoding'])
        if encoding is None:
            continue
        if isinstance(encoding, np.ndarray):
            encoding = encoding.flatten()
        row = {'employee_id': str(emp['_id']), 'name': emp.get('name', 'Unknown'), 'department': emp.get('department', ''), 'employee_code': emp.get('employee_code', '')}
        for i, val in enumerate(encoding):
            row[f'feature_{i}'] = val
        row['label'] = str(emp['_id'])
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(DATASET_DIR, f'face_dataset_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    return csv_path


def prepare_training_data(employees_list):
    X, y = [], []
    name_map = {}
    for emp in employees_list:
        if not emp.get('face_encoding'):
            continue
        encoding = decode_encoding(emp['face_encoding'])
        if encoding is None:
            continue
        if isinstance(encoding, np.ndarray):
            encoding = encoding.flatten()
        X.append(encoding)
        emp_id_str = str(emp['_id'])
        y.append(emp_id_str)
        name_map[emp_id_str] = emp.get('name', 'Unknown')
    if len(X) < 2:
        return None, None, None, "Need at least 2 employees with face data"
    X = np.array(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le, name_map


def augment_face_data(X, y, num_variations=5, noise_factor=0.01):
    X_aug, y_aug = X.copy(), y.copy()
    rng = np.random.RandomState(42)
    for i in range(len(X)):
        original = X[i]
        label = y[i]
        for _ in range(num_variations):
            noise = rng.normal(0, noise_factor, original.shape)
            augmented = original + noise
            augmented = augmented / np.linalg.norm(augmented) if np.linalg.norm(augmented) > 0 else augmented
            X_aug = np.vstack([X_aug, augmented.reshape(1, -1)])
            y_aug = np.append(y_aug, label)
    return X_aug, y_aug


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=15, min_samples_split=4, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_all_models(employees_list, augment=True):
    X, y_encoded, le, name_map = prepare_training_data(employees_list)
    if X is None:
        return {'error': 'Need at least 2 employees with registered face data'}
    n_classes = len(np.unique(y_encoded))
    n_samples = len(X)
    if augment and n_samples < 10:
        X, y_encoded = augment_face_data(X, y_encoded, num_variations=max(3, 10 // n_samples))
    if len(X) < 5 or n_classes < 2:
        return {'error': f'Need more data. Have {n_classes} classes, {len(X)} samples.'}
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    results = {}
    models = {}
    trainers = [
        ('Random Forest', train_random_forest),
        ('SVM (RBF)', train_svm),
        ('Decision Tree', train_decision_tree)
    ]
    for name, trainer_fn in trainers:
        try:
            model = trainer_fn(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            try:
                cv_scores = cross_val_score(model, X, y_encoded, cv=min(3, n_classes))
                cv_mean = cv_scores.mean()
            except Exception:
                cv_mean = acc
            results[name] = {
                'accuracy': round(float(acc) * 100, 2),
                'precision': round(float(prec) * 100, 2),
                'recall': round(float(rec) * 100, 2),
                'f1_score': round(float(f1) * 100, 2),
                'cv_mean': round(float(cv_mean) * 100, 2),
                'train_samples': int(len(X_train)),
                'test_samples': int(len(X_test))
            }
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            results[name]['classification_report'] = report
            models[name] = model
        except Exception as e:
            results[name] = {'error': str(e)}
    best_model_name = max(results, key=lambda k: results[k].get('accuracy', 0))
    joblib.dump(models.get('Random Forest'), RF_MODEL_PATH)
    joblib.dump(models.get('SVM (RBF)'), SVM_MODEL_PATH)
    joblib.dump(models.get('Decision Tree'), DT_MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    meta = {
        'best_model': best_model_name,
        'results': results,
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'n_classes': n_classes,
        'n_samples_total': int(len(X)),
        'n_samples_augmented': int(len(X)),
        'name_map': name_map,
        'original_samples': n_samples
    }
    joblib.dump(meta, TRAINING_META_PATH)
    csv_path = generate_csv_dataset(employees_list)
    meta['dataset_path'] = csv_path
    return meta


def load_trained_models():
    models = {}
    paths = {
        'Random Forest': RF_MODEL_PATH,
        'SVM (RBF)': SVM_MODEL_PATH,
        'Decision Tree': DT_MODEL_PATH
    }
    for name, path in paths.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception:
                models[name] = None
        else:
            models[name] = None
    le = None
    if os.path.exists(LABEL_ENCODER_PATH):
        try:
            le = joblib.load(LABEL_ENCODER_PATH)
        except Exception:
            le = None
    meta = None
    if os.path.exists(TRAINING_META_PATH):
        try:
            meta = joblib.load(TRAINING_META_PATH)
        except Exception:
            meta = None
    return models, le, meta


def predict_with_models(encoding_b64):
    encoding = decode_encoding(encoding_b64)
    if encoding is None:
        return None
    if isinstance(encoding, np.ndarray):
        encoding = encoding.flatten().reshape(1, -1)
    models, le, meta = load_trained_models()
    if le is None or meta is None:
        return None
    predictions = {}
    for model_name, model in models.items():
        if model is not None:
            try:
                pred_id = model.predict(encoding)[0]
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(encoding)[0]
                    confidence = float(max(probs) * 100)
                else:
                    confidence = float(50.0)
                predicted_label = le.inverse_transform([pred_id])[0]
                predictions[model_name] = {
                    'predicted_id': predicted_label,
                    'confidence': round(confidence, 2)
                }
            except Exception as e:
                predictions[model_name] = {'error': str(e)}
        else:
            predictions[model_name] = {'error': 'Model not trained'}
    return predictions


def get_model_accuracy_history():
    if os.path.exists(TRAINING_META_PATH):
        try:
            return joblib.load(TRAINING_META_PATH)
        except Exception:
            return None
    return None

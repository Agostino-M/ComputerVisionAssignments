import os
import json
import pickle
import random
import argparse
import cv2
import dlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from tqdm import tqdm
from joblib import dump, load

def process_real_uncropped_imgs(base_path_real_imgs, output_real_crop_dir):
    dataset_entries = []
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    dirs = os.listdir(base_path_real_imgs)
    for subject_folder in tqdm(dirs, total=len(dirs), desc="Loading and cropping real face images"):
        subject_path = os.path.join(base_path_real_imgs, subject_folder)
        if not os.path.isdir(subject_path):
            print("Skipped", subject_path)
            continue

        for img_file in os.listdir(subject_path):
            img_path = os.path.join(subject_path, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("Skipped", img_path)
                continue

            try:
                faces = face_detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100))

                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    face_crop = image[y:y+h, x:x+w]

                    # Create subject subfolder if it doesn't exist
                    save_dir = os.path.join(output_real_crop_dir, subject_folder)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, img_file)
                    cv2.imwrite(save_path, face_crop)

                    dataset_entries.append({
                        "filepath": save_path,
                        "label": 0,
                        "subject_id": subject_folder,
                    })
            except Exception as e:
                print(f"[WARNING] Failed to process image {img_path}: {e}")
    
    print(f"[INFO] Processed real uncropped images: {len(dataset_entries)} usable images")
    return dataset_entries

def align_face(image, predictor, face_rect):
    landmarks = predictor(image, face_rect)

    # Get mean of eyes landmarks (index 36-41 left, 42-47 right)
    left_eye = np.mean([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)], axis=0)
    right_eye = np.mean([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)], axis=0)

    # Get angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    center = ((face_rect.left() + face_rect.right()) // 2, (face_rect.top() + face_rect.bottom()) // 2)

    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return aligned

def apply_preprocessing_filter(image, filter_config):
    if not filter_config or filter_config["type"] == "none":
        return image

    ftype = filter_config["type"]
    params = filter_config.get("params", {})

    if ftype == "gaussian":
        #print(f"[INFO] Gaussian filter found in {filter_config}")
        ksize = params.get("ksize", 3)
        sigma = params.get("sigma", 0)
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    elif ftype == "median":
        #print(f"[INFO] Median filter found in {filter_config}")
        ksize = params.get("ksize", 3)
        return cv2.medianBlur(image, ksize)

    elif ftype == "bilateral":
        #print(f"[INFO] Bilateral filter found in {filter_config}")
        d = params.get("d", 5)
        sigmaColor = params.get("sigmaColor", 75)
        sigmaSpace = params.get("sigmaSpace", 75)
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    else:
        raise ValueError(f"Unsupported filter type: {ftype}")

def extract_dataset_features(df_dataset, output_path, P, R, n_bins, range_max, method, config_filter):
    #print(len(df))
    features = []
    
    print("[DEBUG] Extracting features...")
    for _, row in tqdm(df_dataset.iterrows(), total=len(df_dataset), desc=f"Extracting LBP ({method}, P={P}, R={R}, bins={n_bins})"):
        try:
            #print(f"Extracting {row['filepath']}")
            image = cv2.imread(row["filepath"], cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Image not found or bad: {row['filepath']}")
            
            filtered_img = apply_preprocessing_filter(image, config_filter)

            lbp = local_binary_pattern(filtered_img, P, R, method)
            #print(P, n_bins)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, range_max), density=True)

            features.append({
                "features": hist,
                "label": row["label"],
                "subject_id": row["subject_id"]
            })
        except Exception as e:
            print(f"[WARNING] Cannot process {row['filepath']}: {e}")
            raise e
    
    with open(output_path, "wb") as f:
        pickle.dump(features, f)
    
    print(f"[RESULT] Extracting features done, saved file with {len(features)} vectors in '{output_path}'")

def preparing_dataset(base_path_fake_imgs, base_path_real_imgs, metadata_csv, output_real_crop_dir, lbp_configs, filter_config):
    print("[DEBUG] Preparing dataset...")
    dataset_entries = []

    # Loaind cropped fake images to dataset
    dirs = os.listdir(base_path_fake_imgs)
    for subject_folder in tqdm(dirs, total=len(dirs), desc="Loading fake face images"):
        subject_path = os.path.join(base_path_fake_imgs, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        for img_file in os.listdir(subject_path):
            dataset_entries.append({
                "filepath": os.path.join(subject_path, img_file),
                "label": 1,
                "subject_id": subject_folder,
            })

    print(f"[INFO] Loaded fake face images: {len(dataset_entries)}")
    # Loaind cropped real images to dataset
    dataset_entries.extend(process_real_uncropped_imgs(base_path_real_imgs, output_real_crop_dir))
    df_dataset = pd.DataFrame(dataset_entries)
    df_dataset.to_csv(metadata_csv, index=False)
    print(f"[DEBUG] Dataset saved to '{metadata_csv}' with {len(df_dataset)} entries.")
    
    lbp_configs = set([(x["lbp_method"], x["P"], x["R"], x["bins"], x["range_max"]) for x in lbp_configs])
    for config in lbp_configs:
        print(f"[INFO] lbp_configs: {config}")
        method, P, R, n_bins, range_max = config
        features_pkl = f"features_{method}_{P}_{R}_{n_bins}.pkl"
        
        extract_dataset_features(df_dataset, features_pkl, P, R, n_bins, range_max, method, filter_config)

def evaluate_all_models(configurations, splits, classifiers, models_xlsx):
    all_models = []
    config_to_features = {}
    for i, config in enumerate(configurations):
        print("-" * 70)
        print(f"[INFO] config {i+1}: {config}")

        lbp_method, do_scale = config["lbp_method"], config["do_scale"]
        P, R, n_bins = config["P"], config["R"], config["bins"]

        features_pkl = f"features_{lbp_method}_{P}_{R}_{n_bins}.pkl"

        print(f"[INFO] data percentage splits: {splits}")
        p_train, p_test, p_val = splits["p_train"], splits["p_test"], splits["p_val"]

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_subjects_train_test_val(features_pkl, p_train, p_test, p_val)

        config_id = f"{lbp_method}_{'scale' if do_scale else 'not_scale'}"
        models_output_csv = f"lbp-{config_id}.csv"

        config_to_features[config_id] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "features_pkl": features_pkl,
        }

        results = model_selection(
            X_train, y_train, X_val, y_val,
            classifiers, do_scale,
            lbp_method, P, R,
            models_output_csv
        )
        for model in results:
            model["config_id"] = config_id
            model["lbp_config"] = config
            all_models.append(model)

    #print(all_models)
    save_model_selection_output(all_models, models_xlsx)
    return all_models, config_to_features

def get_X_y_from_subject_list(data: pickle, subject_list: list):
    entries = [entry for entry in data if entry["subject_id"] in subject_list]
    X = np.array([entry["features"] for entry in entries])
    y = np.array([entry["label"] for entry in entries])
    return X, y

def split_subjects_train_test_val(features_path, p_train, p_test, p_val):
    assert p_train + p_test + p_val == 1, "Total probability must be 1"

    print(f"[INFO] Reading features from pickle {features_path}")
    with open(features_path, "rb") as f:
        data = pickle.load(f)
    print(f"[RESULT] Loaded {len(data)} samples")
    #print("Example entry:", data[0])

    all_subjects = list(set([row["subject_id"] for row in data]))
    n = len(all_subjects)
    n_train = int(n * p_train)
    n_val = int(n * p_val)
    n_test = n - (n_train + n_val)

    random.shuffle(all_subjects)

    train_subjects = all_subjects[:n_train]
    val_subjects = all_subjects[n_train : n_train + n_val]
    test_subjects = all_subjects[n_train + n_val :]

    X_train, y_train = get_X_y_from_subject_list(data, train_subjects)
    X_test, y_test = get_X_y_from_subject_list(data, test_subjects)
    X_val, y_val = get_X_y_from_subject_list(data, val_subjects)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train_and_validate(X_train, y_train, X_val, y_val, classifier, do_scale=False, lbp_P=None, lbp_R=None):
    if do_scale:
        print("[INFO] Using StandardScaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    classifier.fit(X_train, y_train)
    y_val_pred = classifier.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    return {
        "classifier": classifier.__class__.__name__,
        "model": classifier,
        "scaling": do_scale,
        "val_accuracy": val_acc,
        "P": lbp_P,
        "R": lbp_R,
    }

def save_results(all_results, filepath):
    df = pd.DataFrame([{
        "classifier": r["classifier"],
        "scaling": r["scaling"],
        "val_accuracy": r["val_accuracy"],
    } for r in all_results])
    df.to_csv(filepath, index=False)

def model_selection(X_train, y_train, X_val, y_val, classifiers, do_scale, lbp_method, lbp_P, lbp_R, output_filepath):
    classifier_map = {
        "random_forest": RandomForestClassifier(min_samples_leaf=2, max_features="sqrt", random_state=42),
        "logistic_regression": LogisticRegression(random_state=42),
        "linear_svc": LinearSVC(random_state=42),
    }

    all_results = []
    for c in classifiers:
        result = train_and_validate(
            X_train, y_train, X_val, y_val,
            classifier_map[c], do_scale,
            lbp_P=lbp_P, lbp_R=lbp_R
        )
        result["lbp_method"] = lbp_method
        all_results.append(result)
        print(f"[RESULT] Classifier {result['classifier']} (scaling={result['scaling']}, P={lbp_P}, R={lbp_R}) VAL_ACC={result['val_accuracy']:.4f}")

    save_results(all_results, output_filepath)
    return all_results

def save_model_selection_output(all_models, output_filepath="model_selection.xlsx"):
    df = pd.DataFrame(all_models)

    df = df.pivot_table(
        index=['lbp_method', 'P', 'R', 'scaling'],
        columns='classifier',
        values='val_accuracy',
        aggfunc='first'
    ).reset_index()
    df.columns.name = None
    df.to_excel(output_filepath)
    print(f"[RESULT] Models Output saved: '{output_filepath}'")

def evaluate_best_model(best_model, config_to_features):
    best_model_features = config_to_features[best_model["config_id"]]

    X_train = best_model_features["X_train"]
    y_train = best_model_features["y_train"]
    X_test = best_model_features["X_test"]
    y_test = best_model_features["y_test"]
    P = best_model["P"]
    R = best_model["R"]
    
    print(f"[INFO] Evaluating with test data the best model with LBP method={best_model['lbp_method']}, P={P}, R={R}")

    if best_model['scaling']:
        print("[INFO] Using StandardScaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        best_model['scaler'] = scaler

    best_model['model'].fit(X_train, y_train)
    y_test_pred = best_model['model'].predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)

    print(f"[RESULT] Test accuracy: {test_acc:.4f}")
    print(f"[RESULT] Confusion Matrix:\n{cm}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Best Model")
    plt.show()

def compute_pipeline(cfg):
    base_path_fake_imgs = cfg["paths"]["fake"] 
    base_path_real_imgs = cfg["paths"]["real"]
    output_real_crop_dir = cfg["paths"]["output_real_crop_dir"]

    metadata_csv = cfg["output"]["metadata_csv"]
    models_xlsx = cfg["output"]["models_xlsx"]
    classifiers = cfg["classifiers"]
    configurations = cfg["configurations"]
    filter_config = cfg["filter"]
    splits = cfg["splits"]

    best_model_file_model = cfg["output"]["best_model_file_model"]
    best_model_file_scaler = cfg["output"]["best_model_file_scaler"]
    best_model_config_json = cfg["output"]["best_model_config_json"]

    if all(set([os.path.isfile(f"features_{c['lbp_method']}_{c['P']}_{c['R']}_{c['bins']}.pkl") for c in configurations])):
        print("[INFO] pickles found, using cached version")
    else:
        print("[INFO] pickles not found")
        preparing_dataset(base_path_fake_imgs, base_path_real_imgs, metadata_csv, output_real_crop_dir, configurations, filter_config)

    all_models, config_to_features = evaluate_all_models(configurations, splits, classifiers, models_xlsx)
    best_model = max(all_models, key=lambda r: r["val_accuracy"])
    #print(f"[DEBUG] Best model is: {best_model}")
    print(f"[RESULT] Best model is lbp_{best_model['lbp_config']['lbp_method']} {best_model['classifier']} (scaling={best_model['scaling']}) VAL_ACC={best_model['val_accuracy']:.4f}")
    evaluate_best_model(best_model, config_to_features)

    # Salva il modello
    dump(best_model["model"], best_model_file_model)

    # Salva lo scaler, se esiste
    if best_model["scaling"] and "scaler" in best_model:
        dump(best_model["scaler"], best_model_file_scaler)

    # Salva la configurazione migliore
    best_model_config = {
        "lbp_method": best_model["lbp_config"]["lbp_method"],
        "P": best_model["lbp_config"]["P"],
        "R": best_model["lbp_config"]["R"],
        "bins": best_model["lbp_config"]["bins"],
        "range_max": best_model["lbp_config"]["range_max"],
        "classifier": best_model["classifier"],
        "do_scale": best_model["scaling"]
    }
    with open(best_model_config_json, "w") as f:
        json.dump(best_model_config, f)


def do_inference(image, cfg):
    best_model_file_model = cfg["output"]["best_model_file_model"]
    best_model_file_scaler = cfg["output"]["best_model_file_scaler"]
    best_model_config_json = cfg["output"]["best_model_config_json"]

    # Load model, scaler and config
    with open(best_model_config_json) as f:
        config = json.load(f)

    model = load(best_model_file_model)
    scaler = load(best_model_file_scaler) if config["do_scale"] else None

    # Read and process image
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100))
    if len(faces) == 0:
        print("[ERROR] Couldn't detect a face in the image")
        return
    
    filter_config = config.get("filter", {"type": "none"})
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    for i, (x, y, w, h) in enumerate(faces):
        im_copy = image.copy()
        # Align face
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        aligned_img = align_face(im_copy, predictor, dlib_rect)
        face_crop = aligned_img[y:y+h, x:x+w]

        # Apply filter
        filtered_img = apply_preprocessing_filter(face_crop, filter_config)

        # Extract LBP features
        lbp = local_binary_pattern(filtered_img, P=config["P"], R=config["R"], method=config["lbp_method"])
        hist, _ = np.histogram(lbp.ravel(), bins=config["bins"], range=(0, config["range_max"]), density=True)
        X_input = hist.reshape(1, config["bins"])

        if scaler:
            X_input = scaler.transform(X_input)

        # Predict
        y_pred = model.predict(X_input)[0]
        label_str = "REAL" if y_pred == 0 else "FAKE"
        print(f"[RESULT] Face {i+1}/{len(faces)}: Predicted class: {label_str}")

        # Draw rectangle and label
        color = (0, 255, 0) if y_pred == 0 else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label_str, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show result
    cv2.imshow("Risultato", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake Face Detection")
    parser.add_argument("--config", type=str, default="default_config.json", help="Path to JSON config file")
    parser.add_argument("--path", type=str, help="Path to image file to classify")
    parser.add_argument("--force_pipeline", action="store_true", help="Force calculation of best model")
    args = parser.parse_args()
    
    print("Started with following parameters:")
    print(f"\t - Config: {args.config}")
    print(f"\t - Path: {args.path}")
    print(f"\t - Force pipeline attivo: {args.force_pipeline}")

    print(f"[INFO] JSON config file: {args.config}")
    # Load custom configurations
    with open(args.config) as f:
        cfg = json.load(f)

    if args.force_pipeline or not os.path.isfile(cfg["output"]["best_model_file_model"]):
        print("[INFO] Recalculating pipeline to find best model")
        compute_pipeline(cfg)

    if args.path:
        image = cv2.imread(args.path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Can't read image")
        print(f"[INFO] Inference on file: {args.path}")
        do_inference(image, cfg)
    else:
        print("[DEBUG] No image path found to classify")
    

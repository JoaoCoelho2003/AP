import numpy as np
import tensorflow as tf
import pandas as pd
import os
import re
import pickle
from nltk.tokenize import word_tokenize
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import csv
import string

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

nltk.download("punkt", quiet=True)


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s.,!?]", "", text)
    tokens = word_tokenize(text)
    return " ".join(tokens)


def extract_features(text):
    if not isinstance(text, str):
        return [0, 0, 0, 0]

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sentence_length = (
        np.mean([len(s.split()) for s in sentences]) if sentences else 0
    )

    words = re.findall(r"\b\w+\b", text.lower())
    lexical_diversity = len(set(words)) / len(words) if words else 0

    punctuation_count = sum(1 for char in text if char in string.punctuation)
    punctuation_freq = punctuation_count / len(text) if text else 0

    first_person = len(
        re.findall(
            r"\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b", text, re.IGNORECASE
        )
    )
    first_person_freq = first_person / len(words) if words else 0

    return [avg_sentence_length, lexical_diversity, punctuation_freq, first_person_freq]


def load_model(model_type="lstm"):
    model_path = f"trained_models/tensorflow/{model_type}_model.h5"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded {model_type} model")
        return model
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        return None


def load_tokenizer_and_metadata():
    try:
        if os.path.exists("improved_data/tokenizer.pkl"):
            with open("improved_data/tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)

            with open("improved_data/metadata.pkl", "rb") as f:
                metadata = pickle.load(f)

            print("Loaded tokenizer and metadata from improved_data")
            return tokenizer, metadata

        elif os.path.exists("preprocessed_tf/tokenizer.pkl"):
            with open("preprocessed_tf/tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)

            with open("preprocessed_tf/metadata.pkl", "rb") as f:
                metadata = pickle.load(f)

            print("Loaded tokenizer and metadata from preprocessed_tf")
            return tokenizer, metadata

        else:
            print("Could not find tokenizer and metadata files")
            return None, None

    except Exception as e:
        print(f"Error loading tokenizer and metadata: {e}")
        return None, None


def load_datasets():
    inputs_path = "./datasets/clean/teacher_inputs.csv"
    outputs_path = "./datasets/clean/teacher_outputs.csv"

    try:
        print(f"Loading input texts from {inputs_path}...")
        inputs_data = []
        input_ids = []

        try:
            df_inputs = pd.read_csv(inputs_path)
            if "ID" in df_inputs.columns and "Text" in df_inputs.columns:
                for _, row in df_inputs.iterrows():
                    input_ids.append(row["ID"])
                    inputs_data.append(row["Text"])
                print(
                    f"Loaded {len(inputs_data)} entries from CSV with ID and Text columns"
                )
            else:
                raise ValueError("CSV doesn't have expected columns")
        except:
            with open(inputs_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if row and len(row) > 0:
                        if re.match(r"D\d+-\d+", row[0]):
                            id_val = row[0]
                            text = ",".join(row[1:]) if len(row) > 1 else ""
                        else:
                            id_val = f"row-{i}"
                            text = ",".join(row)

                        input_ids.append(id_val)
                        inputs_data.append(text)

        print(f"Loading output labels from {outputs_path}...")
        outputs_data = {}

        try:
            df_outputs = pd.read_csv(outputs_path, sep="\t")
            if "ID" in df_outputs.columns and "Label" in df_outputs.columns:
                for _, row in df_outputs.iterrows():
                    outputs_data[row["ID"]] = 1 if row["Label"] == "AI" else 0
                print(
                    f"Loaded {len(outputs_data)} labels from CSV with ID and Label columns"
                )
            else:
                raise ValueError("CSV doesn't have expected columns")
        except:
            with open(outputs_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split("\t")
                    if len(parts) >= 2:
                        id_val = parts[0].strip()
                        label = parts[1].strip()
                        outputs_data[id_val] = 1 if label == "AI" else 0

        if len(input_ids) > 0 and all(
            id_val not in outputs_data for id_val in input_ids[:10]
        ):
            print(
                "No matching IDs found in outputs. Trying to extract IDs from text..."
            )
            new_input_ids = []
            new_inputs_data = []

            for i, text in enumerate(inputs_data):
                id_match = re.search(r"(D\d+-\d+)", text)
                if id_match:
                    id_val = id_match.group(1)
                    text_without_id = re.sub(r"D\d+-\d+\s*", "", text, 1)
                    new_input_ids.append(id_val)
                    new_inputs_data.append(text_without_id)
                else:
                    new_input_ids.append(input_ids[i])
                    new_inputs_data.append(text)

            input_ids = new_input_ids
            inputs_data = new_inputs_data

        dataset = []
        processed_count = 0

        for i, (id_val, text) in enumerate(zip(input_ids, inputs_data)):
            if id_val in outputs_data:
                dataset.append((id_val, text, outputs_data[id_val]))
                processed_count += 1

        print(f"Successfully matched {processed_count} entries with labels")
        if processed_count == 0:
            print("WARNING: No entries could be matched with labels!")
            if abs(len(inputs_data) - len(outputs_data)) < 10:
                print("Attempting direct index matching as a last resort...")
                output_ids = list(outputs_data.keys())
                dataset = []
                for i in range(min(len(inputs_data), len(output_ids))):
                    dataset.append(
                        (output_ids[i], inputs_data[i], outputs_data[output_ids[i]])
                    )
                print(f"Matched {len(dataset)} entries by index")

        return dataset

    except Exception as e:
        print(f"Error loading datasets: {e}")
        import traceback

        traceback.print_exc()
        return []


def predict_with_model(model, text, tokenizer, metadata, model_type="lstm"):
    if model is None or tokenizer is None or metadata is None:
        return None, None

    max_seq_length = metadata["max_seq_length"]

    if model_type == "dnn":
        features = np.array([extract_features(text)])
        prediction = model.predict(features, verbose=0)[0][0]
        binary_prediction = 1 if prediction >= 0.5 else 0
        return binary_prediction, prediction

    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(
        sequence, maxlen=max_seq_length, padding="post", truncating="post"
    )

    if model_type == "ensemble":
        num_inputs = len(model.inputs)

        inputs = []
        for i in range(num_inputs):
            input_shape = model.inputs[i].shape

            if len(input_shape) == 2 and input_shape[1] == 4:
                inputs.append(np.array([extract_features(text)]))
            else:
                inputs.append(padded_sequence)

        prediction = model.predict(inputs, verbose=0)[0][0]
    else:
        prediction = model.predict(padded_sequence, verbose=0)[0][0]

    binary_prediction = 1 if prediction >= 0.5 else 0

    return binary_prediction, prediction


def evaluate_models():
    tokenizer, metadata = load_tokenizer_and_metadata()
    if tokenizer is None or metadata is None:
        print("Cannot proceed without tokenizer and metadata")
        return

    dataset = load_datasets()
    if not dataset:
        print("Cannot proceed without dataset")
        return

    model_types = ["lstm", "gru", "transformer", "dnn", "ensemble"]
    models = {}

    for model_type in model_types:
        model = load_model(model_type)
        if model is not None:
            models[model_type] = model

    if not models:
        print("No models could be loaded")
        return

    results = {
        model_type: {"correct": 0, "total": 0, "predictions": []}
        for model_type in models
    }

    print("\nEvaluating models on dataset...")
    for id_val, text, true_label in tqdm(dataset):
        cleaned_text = clean_text(text)

        for model_type, model in models.items():
            binary_prediction, raw_prediction = predict_with_model(
                model, cleaned_text, tokenizer, metadata, model_type
            )

            if binary_prediction is None:
                continue

            results[model_type]["total"] += 1
            if binary_prediction == true_label:
                results[model_type]["correct"] += 1

            results[model_type]["predictions"].append(
                {
                    "id": id_val,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "true_label": "AI" if true_label == 1 else "Human",
                    "predicted_label": "AI" if binary_prediction == 1 else "Human",
                    "confidence": (
                        raw_prediction if binary_prediction == 1 else 1 - raw_prediction
                    ),
                    "correct": binary_prediction == true_label,
                }
            )

    print("\n=== Model Evaluation Results ===")
    for model_type in models:
        correct = results[model_type]["correct"]
        total = results[model_type]["total"]
        accuracy = correct / total if total > 0 else 0

        print(f"\n{model_type.upper()} Model:")
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

        human_correct = sum(
            1
            for p in results[model_type]["predictions"]
            if p["true_label"] == "Human" and p["correct"]
        )
        human_total = sum(
            1 for p in results[model_type]["predictions"] if p["true_label"] == "Human"
        )
        human_accuracy = human_correct / human_total if human_total > 0 else 0

        ai_correct = sum(
            1
            for p in results[model_type]["predictions"]
            if p["true_label"] == "AI" and p["correct"]
        )
        ai_total = sum(
            1 for p in results[model_type]["predictions"] if p["true_label"] == "AI"
        )
        ai_accuracy = ai_correct / ai_total if ai_total > 0 else 0

        print(f"Human accuracy: {human_accuracy:.4f} ({human_correct}/{human_total})")
        print(f"AI accuracy: {ai_accuracy:.4f} ({ai_correct}/{ai_total})")

    os.makedirs("evaluation", exist_ok=True)
    for model_type in models:
        with open(f"evaluation/{model_type}_dataset_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["ID", "Text", "True Label", "Predicted Label", "Confidence", "Correct"]
            )

            for p in results[model_type]["predictions"]:
                writer.writerow(
                    [
                        p["id"],
                        p["text"],
                        p["true_label"],
                        p["predicted_label"],
                        f"{p['confidence']:.4f}",
                        "Yes" if p["correct"] else "No",
                    ]
                )

    print("\nDetailed results saved to evaluation/ directory")


if __name__ == "__main__":
    evaluate_models()

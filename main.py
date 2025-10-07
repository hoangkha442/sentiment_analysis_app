import os
from flask import Flask, render_template, request, jsonify
from langdetect import detect
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = Flask(__name__)

# --------- Models ----------
VI_MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"
EN_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Vietnamese model
# vi_tokenizer = AutoTokenizer.from_pretrained(VI_MODEL_NAME, use_fast=False)
# vi_model = AutoModelForSequenceClassification.from_pretrained(VI_MODEL_NAME).to(device)
# vi_model.eval()
vi_tokenizer = AutoTokenizer.from_pretrained(VI_MODEL_NAME, use_fast=False)
vi_model = AutoModelForSequenceClassification.from_pretrained(VI_MODEL_NAME)
vi_model.eval()
sentiment_pipeline = pipeline("sentiment-analysis", model=vi_model, tokenizer=vi_tokenizer)


# English model
en_tokenizer = AutoTokenizer.from_pretrained(EN_MODEL_NAME)
en_model = AutoModelForSequenceClassification.from_pretrained(EN_MODEL_NAME).to(device)
en_model.eval()

# Label mapping cho PhoBERT
vi_label_map = {
    0: ("NEGATIVE", "Tiêu cực"),
    1: ("NEUTRAL", "Trung tính"),
    2: ("POSITIVE", "Tích cực")
}

# Label mapping cho tiếng Anh
en_label_map = {
    0: ("NEGATIVE", "Negative"),
    1: ("POSITIVE", "Positive")
}


# -----------------------------
# Ngôn ngữ nhận diện
# -----------------------------
def detect_lang(text: str) -> str:
    try:
        lang = detect(text)
        if lang.startswith("vi"):
            return "vi"
        elif lang.startswith("en"):
            return "en"
        else:
            if any(ch in text for ch in "ăâđêôơưáàạảãấầậẩẫắằặẳẵéèẹẻẽếềệểễóòọỏõốồộổỗớờợởỡíìịỉĩúùụủũứừựửữýỳỵỷỹ"):
                return "vi"
            return "en"
    except Exception:
        if any(ch in text for ch in "ăâđêôơưáàạảãấầậẩẫắằặẳẵéèẹẻẽếềệểễóòọỏõốồộổỗớờợởỡíìịỉĩúùụủũứừựửữýỳỵỷỹ"):
            return "vi"
        return "en"


# -----------------------------
# Phân tích tiếng Việt (PhoBERT)
# -----------------------------
# def analyze_vi(text: str):
#     inputs = vi_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
#     with torch.no_grad():
#         outputs = vi_model(**inputs)
#         logits = outputs.logits.squeeze(0)
#         probs = torch.softmax(logits, dim=-1)

#     label_idx = int(torch.argmax(probs).item())
#     eng_label, vi_label = vi_label_map[label_idx]
#     confidence = float(probs[label_idx].item())

#     scores = {
#         vi_label_map[i][1]: round(float(probs[i].item()), 3) for i in range(3)
#     }

#     return {
#         "language": "vi",
#         "label": vi_label, 
#         "english_label": eng_label,
#         "score": round(confidence, 3),
#         "scores": scores
#     }

def analyze_vi(text: str):
    if not text.strip():
        return {"error": "Text is empty."}

    # Dùng pipeline của transformers
    result = sentiment_pipeline(text)[0]
    label = result["label"]
    score = round(result["score"], 3)

    # Map nhãn tiếng Việt
    label_map = {
        "POS": "Tích cực",
        "NEG": "Tiêu cực",
        "NEU": "Trung tính"
    }

    vi_label = label_map.get(label, label)

    # Trả kết quả tương thích với frontend
    return {
        "language": "vi",
        "label": vi_label,
        "english_label": label,  # Giữ nhãn gốc POS/NEG/NEU
        "score": score,
        "scores": {
            "Tích cực": score if label == "POS" else 0.0,
            "Trung tính": score if label == "NEU" else 0.0,
            "Tiêu cực": score if label == "NEG" else 0.0
        }
    }
# -----------------------------
# Phân tích tiếng Anh
# -----------------------------
def analyze_en(text: str):
    inputs = en_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = en_model(**inputs)
        logits = outputs.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)

    label_idx = int(torch.argmax(probs).item())
    eng_label, vi_label = en_label_map[label_idx]
    confidence = float(probs[label_idx].item())

    scores = {
        en_label_map[i][1]: round(float(probs[i].item()), 3) for i in range(2)
    }

    return {
        "language": "en",
        "label": vi_label,  # Giữ English, có thể đổi sang tiếng Việt nếu muốn
        "english_label": eng_label,
        "score": round(confidence, 3),
        "scores": scores
    }


# -----------------------------
# Flask routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    lang = (data.get("lang") or "auto").lower()
    if not text:
        return jsonify({"error": "Text is empty."}), 400

    if lang == "auto":
        lang = detect_lang(text)

    if lang == "vi":
        result = analyze_vi(text)
    else:
        result = analyze_en(text)

    return jsonify({
        "ok": True,
        "input": {"text": text, "lang": lang},
        "result": result
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)

# import os
# from flask import Flask, render_template, request, jsonify
# from langdetect import detect
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# import torch
# from pyngrok import ngrok

# # ✅ Nếu chạy local, tạo cache tạm an toàn
# os.environ["HF_HOME"] = "./hf_cache"
# os.makedirs("./hf_cache", exist_ok=True)

# app = Flask(__name__)

# # --------- Models ----------
# VI_MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"
# EN_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# device = "cuda" if torch.cuda.is_available() else "cpu"

# print("🔄 Loading Vietnamese model...")
# vi_tokenizer = AutoTokenizer.from_pretrained(VI_MODEL_NAME, use_fast=False)
# vi_model = AutoModelForSequenceClassification.from_pretrained(VI_MODEL_NAME).to(device)
# vi_model.eval()
# vi_pipeline = pipeline("sentiment-analysis", model=vi_model, tokenizer=vi_tokenizer)
# print("✅ PhoBERT loaded successfully.")

# print("🔄 Loading English model...")
# en_tokenizer = AutoTokenizer.from_pretrained(EN_MODEL_NAME)
# en_model = AutoModelForSequenceClassification.from_pretrained(EN_MODEL_NAME).to(device)
# en_model.eval()
# print("✅ English model loaded successfully.")

# def detect_lang(text):
#     try:
#         lang = detect(text)
#         if lang.startswith("vi"):
#             return "vi"
#         elif lang.startswith("en"):
#             return "en"
#     except:
#         pass
#     return "vi" if any(ch in text for ch in "ăâđêôơưáàạảãấầậẩẫắằặẳẵéèẹẻẽếềệểễóòọỏõốồộổỗớờợởỡíìịỉĩúùụủũứừựửữýỳỵỷỹ") else "en"

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/analyze", methods=["POST"])
# def analyze():
#     data = request.get_json(force=True)
#     text = data.get("text", "").strip()
#     lang = data.get("lang", "auto")
#     if not text:
#         return jsonify({"error": "Empty input"}), 400

#     if lang == "auto":
#         lang = detect_lang(text)

#     if lang == "vi":
#         result = vi_pipeline(text)[0]
#         label_map = {"POS": "Tích cực", "NEG": "Tiêu cực", "NEU": "Trung tính"}
#         label = label_map.get(result["label"], result["label"])
#         score = round(result["score"], 3)
#         output = {"lang": "vi", "label": label, "score": score}
#     else:
#         inputs = en_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
#         with torch.no_grad():
#             outputs = en_model(**inputs)
#             probs = torch.softmax(outputs.logits, dim=-1)
#         idx = int(torch.argmax(probs))
#         label = ["Negative", "Positive"][idx]
#         score = round(float(probs[0][idx]), 3)
#         output = {"lang": "en", "label": label, "score": score}

#     return jsonify({"ok": True, "result": output})

# # -----------------------------
# # Run Flask + ngrok
# # -----------------------------
# if __name__ == "__main__":
#     from pyngrok import ngrok

#     port = 7860
#     public_url = ngrok.connect(port).public_url
#     print(f"🌍 Public URL: {public_url}")
#     app.run(port=port)

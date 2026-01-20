from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
import pytesseract
from PIL import Image
import os
import torch

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Import legal section data
from sections_datapart1 import sections_datapart1
from sections_datapart2 import sections_datapart2

app = Flask(__name__)

sections_data = sections_datapart1 + sections_datapart2

# Load Legal-BERT
print("Loading Legal-BERT model...")
model = SentenceTransformer("nlpaueb/legal-bert-small-uncased")
print("✅ Legal-BERT loaded!")

# Precompute embeddings
print("Encoding legal corpus...")
corpus = [
    section['title'] + ". " +
    section['description'] + " " +
    " ".join(section.get('keywords', []))
    for section in sections_data
]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
print("✅ Corpus embeddings ready!")

# Multilingual crime keywords
crime_mapping = {
    "theft": ["theft", "stealing", "robbery", "चोरी", "திருட்டு", "దొంగతనం", "മോഷണം"],
    "murder": ["murder", "homicide", "हत्या", "கொலை", "హత్య", "കൊല"],
    "rape": ["rape", "assault", "बलात्कार", "வன்புணர்ச்சி", "బలాత్కారం", "ബലാത്സംഗം"],
    "fraud": ["fraud", "scam", "धोखाधड़ी", "மோசடி", "మోసం", "വഞ്ചന"],
    "cybercrime": ["cyber", "hack", "साइबर", "இணைய", "సైబర్", "സൈബർ"]
}

# ---------- Helper Functions ----------

def safe_translate(text):
    if len(text.split()) == 1:
        text = "अपराध " + text
    return GoogleTranslator(source='auto', target='en').translate(text)


def detect_unicode_language(text):
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            return 'hi'
        elif 0x0B80 <= code <= 0x0BFF:
            return 'ta'
        elif 0x0C00 <= code <= 0x0C7F:
            return 'te'
        elif 0x0D00 <= code <= 0x0D7F:
            return 'ml'
    return 'en'


def reverse_translate_results(results, lang_code):
    translated = []
    for section in results:
        sec = section.copy()
        sec["title"] = GoogleTranslator(source='en', target=lang_code).translate(sec["title"])
        sec["description"] = GoogleTranslator(source='en', target=lang_code).translate(sec["description"])
        sec["punishment"] = GoogleTranslator(source='en', target=lang_code).translate(sec["punishment"])
        sec["court"] = GoogleTranslator(source='en', target=lang_code).translate(sec["court"])
        translated.append(sec)
    return translated


def detect_primary_crime(text):
    for crime, keywords in crime_mapping.items():
        if any(word.lower() in text.lower() for word in keywords):
            return crime

    translated = safe_translate(text).lower()
    for crime, keywords in crime_mapping.items():
        if any(word.lower() in translated for word in keywords):
            return crime

    return None


def fetch_primary_sections(crime_type):
    return [
        s for s in sections_data
        if crime_type and any(crime_type in kw.lower() for kw in s["keywords"])
    ]


def run_semantic_search(text):
    query_embed = model.encode(text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embed, corpus_embeddings)[0]
    top_results = torch.topk(scores, k=5)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        section = sections_data[idx.item()].copy()
        section['score'] = float(score)
        results.append(section)
    return results

# ---------- Routes ----------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search")
def search():
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify([])

    user_lang = detect_unicode_language(query)
    crime_type = detect_primary_crime(query)
    translated_query = safe_translate(query)

    primary = fetch_primary_sections(crime_type)
    semantic = run_semantic_search(translated_query)

    seen = set()
    final = []

    for sec in primary + semantic:
        if sec["section_number"] not in seen:
            seen.add(sec["section_number"])
            final.append(sec)

    if user_lang != "en":
        final = reverse_translate_results(final, user_lang)

    return jsonify(final)


@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    file.save(path)

    img = Image.open(path)
    text = pytesseract.image_to_string(img)

    lang = detect_unicode_language(text)
    crime_type = detect_primary_crime(text)
    translated = safe_translate(text)

    primary = fetch_primary_sections(crime_type)
    semantic = run_semantic_search(translated)

    seen = set()
    final = []

    for sec in primary + semantic:
        if sec["section_number"] not in seen:
            seen.add(sec["section_number"])
            final.append(sec)

    if lang != "en":
        final = reverse_translate_results(final, lang)

    return jsonify({
        "ocr_text": text,
        "translated_text": translated,
        "legal_results": final
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

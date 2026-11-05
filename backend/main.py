
import os
import pickle
from pathlib import Path
from io import BytesIO

import faiss
import torch
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTImageProcessor
from sentence_transformers import SentenceTransformer
from google import genai


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

CHECKPOINT_PATH = BASE_DIR / "model" / "best_model.pth"
INDEX_PATH = BASE_DIR / "rag" / "medical.index"
METADATA_PATH = BASE_DIR / "rag" / "chunks_metadata.pkl"

load_dotenv(PROJECT_DIR / ".env")

GEMINI_MODEL = "gemini-3.5-flash-lite"

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

LABEL2ID = {label: i for i, label in enumerate(DISEASE_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

DEVICE = torch.device("cpu")

app = FastAPI(
    title="MedVision API",
    description="Educational chest X-ray analysis using ViT, FAISS and Gemini",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

print("Loading Vision Transformer...")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=14,
    problem_type="multi_label_classification",
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

print("ViT model loaded successfully!")

print("Loading RAG components...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

index = faiss.read_index(str(INDEX_PATH))

with open(METADATA_PATH, "rb") as file:
    rag_data = pickle.load(file)

all_chunks = rag_data["chunks"]
chunk_metadata = rag_data["metadata"]

if index.ntotal != len(all_chunks):
    raise RuntimeError("FAISS index and chunk metadata do not match.")

print(f"FAISS loaded: {index.ntotal} vectors")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

print("Gemini client initialized!")


def predict_findings(image):
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    predictions = [
        {
            "finding": label,
            "score": round(float(score), 4),
            "percentage": round(float(score) * 100, 2)
        }
        for label, score in zip(DISEASE_LABELS, scores)
    ]

    return sorted(
        predictions,
        key=lambda item: item["score"],
        reverse=True
    )


def retrieve_medical_information(query, top_k=5):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            "text": all_chunks[idx],
            "source": chunk_metadata[idx]["source"],
            "similarity_score": round(float(score), 4)
        })

    return results


def generate_medical_answer(query, retrieved_results):
    if not retrieved_results:
        return "Insufficient information found in the medical knowledge base."

    context = "\n\n".join(
        f"Source: {result['source']}\n"
        f"Information: {result['text']}"
        for result in retrieved_results
    )

    prompt = f"""
You are MedVision, an educational medical information assistant.

Explain the model-ranked chest X-ray findings using only the
medical reference context provided below.

Rules:
1. Do not invent medical facts.
2. State clearly when the context is insufficient.
3. Explain terminology in simple language.
4. Do not diagnose the patient.
5. Do not prescribe medication or treatment.
6. Model scores are not calibrated disease probabilities.
7. Model-ranked findings are not confirmed abnormalities.
8. Mention relevant source filenames.
9. Include an educational-use disclaimer.

MEDICAL CONTEXT:
{context}

QUESTION:
{query}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text or "No explanation was generated."


@app.get("/")
def home():
    return {
        "message": "Welcome to MedVision API",
        "status": "running",
        "model": "Vision Transformer",
        "rag": "FAISS",
        "llm": "Gemini"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "vision_model": "loaded",
        "faiss_vectors": index.ntotal,
        "documents": len(set(
            item["source"] for item in chunk_metadata
        ))
    }


@app.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    if file.content_type not in (
        "image/jpeg", "image/png", "image/jpg"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    image_bytes = await file.read(10 * 1024 * 1024 + 1)

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image must be 10 MB or smaller."
        )

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    predictions = predict_findings(image)

    top_findings = [
        item["finding"] for item in predictions[:3]
    ]

    query = (
        "Explain these model-ranked chest X-ray findings "
        "in simple language without assuming they are "
        "confirmed diagnoses: "
        + ", ".join(top_findings)
    )

    retrieved_results = retrieve_medical_information(
        query,
        top_k=5
    )

    explanation = generate_medical_answer(
        query,
        retrieved_results
    )

    sources = list(dict.fromkeys(
        item["source"] for item in retrieved_results
    ))

    return {
        "status": "success",
        "filename": file.filename,
        "predictions": predictions,
        "top_findings": top_findings,
        "explanation": explanation,
        "sources": sources,
        "disclaimer": (
            "MedVision is an educational research prototype, "
            "not a clinically validated diagnostic tool. "
            "Model scores are not calibrated probabilities. "
            "Consult a qualified healthcare professional."
        )
    }

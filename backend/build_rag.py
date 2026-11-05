
from pathlib import Path
import re
import pickle

import requests
import faiss
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
INDEX_DIR = BASE_DIR / "rag"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

medical_sources = {
    "chest_xray": "https://medlineplus.gov/ency/article/003804.htm",
    "atelectasis": "https://medlineplus.gov/ency/article/000065.htm",
    "pneumothorax": "https://medlineplus.gov/ency/article/000087.htm",
    "pneumonia": "https://medlineplus.gov/ency/article/000093.htm",
    "pneumonia_general": "https://medlineplus.gov/ency/article/000145.htm"
}

additional_documents = {
    "nodule.txt": """
SOURCE: RadiologyInfo - Lung Nodules
URL: https://www.radiologyinfo.org/info/lung-nodules

A pulmonary nodule is a small growth in the lung, generally smaller
than 3 cm. Nodules may be detected incidentally on chest X-rays or
CT scans. Many are benign and may be associated with infections,
scarring, or lymph nodes. Some can be malignant.

Chest CT can help evaluate a nodule's size, shape, location, and
other characteristics. Assessment depends on imaging features,
changes over time, and individual risk factors. A model prediction
alone cannot determine whether a nodule is benign or malignant.
""",
    "infiltration.txt": """
SOURCE: NCBI Bookshelf - A Systematic Approach to Chest Radiographic Analysis
URL: https://www.ncbi.nlm.nih.gov/books/NBK553874/

An infiltrate is a term used to describe an abnormal area of
increased opacity on a chest radiograph. Air-space opacities may
appear when the normal air-filled spaces in the lungs contain
material such as fluid or inflammatory cells.

Infiltrates and air-space opacities can have different causes.
They must be interpreted alongside the patient's clinical
information and other imaging findings. The term does not identify
one specific disease by itself.
""",
    "pleural_thickening.txt": """
SOURCE: NCBI Bookshelf - A Systematic Approach to Chest Radiographic Analysis
URL: https://www.ncbi.nlm.nih.gov/books/NBK553874/

Pleural thickening refers to thickening of the pleural surfaces,
the thin membranes surrounding the lungs. It is one type of
pleural abnormality visible on chest imaging.

Pleural abnormalities may include thickening, plaques,
calcification, air in the pleural space, or fluid in the pleural
space. Imaging findings need to be interpreted in context by
a qualified healthcare professional.
"""
}


def download_documents():
    for name, url in medical_sources.items():
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 MedVisionResearch/1.0"},
                timeout=30
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for element in soup.select(
                "script, style, nav, footer, header, aside, form"
            ):
                element.decompose()

            main = soup.find("main") or soup.find("article") or soup.body

            if main is None:
                raise ValueError("No article content found")

            text = main.get_text(separator="\n", strip=True)

            if len(text.split()) < 100:
                raise ValueError("Extracted article is unexpectedly short")

            content = (
                f"SOURCE: MedlinePlus\n"
                f"TITLE: {name.replace('_', ' ').title()}\n"
                f"URL: {url}\n\n{text}"
            )

            path = DOCUMENTS_DIR / f"{name}.txt"
            path.write_text(content, encoding="utf-8")

            print(f"Downloaded: {path.name}")

        except Exception as error:
            print(f"Failed: {name} - {error}")

    for filename, content in additional_documents.items():
        path = DOCUMENTS_DIR / filename
        path.write_text(content.strip(), encoding="utf-8")
        print(f"Created: {filename}")


def clean_medical_text(text):
    text = re.sub(
        r"(?im)^(SOURCE|TITLE|URL):[^\n]*$",
        "",
        text
    )

    text = re.split(
        r"(?im)^\s*(?:Images|References)\s*$",
        text
    )[0]

    text = re.split(
        r"Review Date|Learn how to cite this page",
        text,
        flags=re.IGNORECASE
    )[0]

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_chunks(text, chunk_size=350, overlap=50):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()

        if not words:
            continue

        if current_words + len(words) > chunk_size and current:
            chunks.append(" ".join(current))

            previous_words = " ".join(current).split()
            overlap_words = previous_words[-overlap:]

            current = [" ".join(overlap_words)]
            current_words = len(overlap_words)

        current.append(sentence)
        current_words += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


def build_index():
    all_chunks = []
    chunk_metadata = []

    files = sorted(DOCUMENTS_DIR.glob("*.txt"))

    print(f"\nMedical documents found: {len(files)}")

    for path in files:
        text = path.read_text(encoding="utf-8")
        cleaned = clean_medical_text(text)
        chunks = create_chunks(cleaned)

        for chunk in chunks:
            if len(chunk.split()) >= 15:
                all_chunks.append(chunk)
                chunk_metadata.append({"source": path.name})

    if not all_chunks:
        raise RuntimeError("No valid medical chunks were generated")

    print(f"Total chunks: {len(all_chunks)}")
    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    print("Generating embeddings...")

    embeddings = embedding_model.encode(
        all_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype(np.float32)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(
        index,
        str(INDEX_DIR / "medical.index")
    )

    with open(INDEX_DIR / "chunks_metadata.pkl", "wb") as file:
        pickle.dump(
            {
                "chunks": all_chunks,
                "metadata": chunk_metadata
            },
            file
        )

    print("\nRAG INDEX CREATED SUCCESSFULLY!")
    print("Documents:", len(files))
    print("Vectors:", index.ntotal)
    print("Dimensions:", dimension)
    print("Index:", INDEX_DIR / "medical.index")
    print("Metadata:", INDEX_DIR / "chunks_metadata.pkl")


if __name__ == "__main__":
    build_index()

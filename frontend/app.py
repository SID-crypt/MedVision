
import streamlit as st
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="MedVision | AI Chest X-Ray Analysis",
    page_icon="🩻",
    layout="wide"
)

st.title("🩻 MedVision")
st.subheader("AI-Powered Chest X-Ray Analysis")
st.caption("Vision Transformer + FAISS RAG + Gemini")

st.warning(
    "MedVision is an educational research prototype, "
    "not a clinically validated diagnostic system. "
    "Do not use its predictions for medical decisions."
)

# Check backend availability
try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=10
    )
    health_response.raise_for_status()
    backend_online = True
except requests.RequestException:
    backend_online = False

with st.sidebar:
    st.header("System Information")

    if backend_online:
        st.success("Backend connected")
        st.write("**Vision model:** ViT")
        st.write("**Vector database:** FAISS")
        st.write("**Language model:** Gemini")
        st.write("**API:** FastAPI")
    else:
        st.error("Backend offline")
        st.info(
            "Start FastAPI with: "
            "python -m uvicorn backend.main:app "
            "--host 127.0.0.1 --port 8000"
        )

    st.divider()
    st.write("**Dataset:** NIH ChestX-ray14")
    st.write("**Model:** ViT Base")
    st.write("**Output classes:** 14")
    st.write("**Test mean AUROC:** 0.7535")
    st.caption(
        "AUROC is a ranking metric, not diagnostic accuracy."
    )

st.divider()

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"],
    help="Upload a de-identified JPG or PNG image (maximum 10 MB)."
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")
    except Exception:
        st.error("Unable to read this image.")
        st.stop()

    left, right = st.columns([1, 1])

    with left:
        st.image(
            image,
            caption="Uploaded chest X-ray",
            width="stretch"
        )

    with right:
        st.markdown("### Image details")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Dimensions:** {image.width} × {image.height}")
        st.write(f"**Size:** {len(image_bytes) / (1024 * 1024):.2f} MB")

        if len(image_bytes) > 10 * 1024 * 1024:
            st.error("The image must be 10 MB or smaller.")
            st.stop()

        analyze = st.button(
            "Analyze X-Ray",
            type="primary",
            disabled=not backend_online,
            use_container_width=True
        )

    if analyze:
        with st.spinner(
            "Running ViT inference, retrieving medical "
            "references, and generating an explanation..."
        ):
            try:
                response = requests.post(
                    f"{API_URL}/analyze",
                    files={
                        "file": (
                            uploaded_file.name,
                            image_bytes,
                            uploaded_file.type or "image/png"
                        )
                    },
                    timeout=240
                )
                response.raise_for_status()
                st.session_state["analysis"] = response.json()
                st.session_state["analysis_file"] = uploaded_file.name
                st.session_state["analysis_bytes"] = image_bytes

            except requests.RequestException as error:
                st.error(f"Analysis failed: {error}")

    if (
        "analysis" in st.session_state
        and st.session_state.get("analysis_file") == uploaded_file.name
        and st.session_state.get("analysis_bytes") == image_bytes
    ):
        result = st.session_state["analysis"]

        st.divider()
        st.header("Analysis Results")

        predictions = result.get("predictions", [])

        if predictions:
            df = pd.DataFrame(predictions)
            df["finding"] = df["finding"].str.replace(
                "_", " ", regex=False
            )

            st.subheader("Top Model-Ranked Findings")

            top_three = df.head(3)
            cols = st.columns(3)

            for col, (_, row) in zip(cols, top_three.iterrows()):
                with col:
                    st.metric(
                        row["finding"],
                        f"{row['percentage']:.2f}%"
                    )

            st.caption(
                "These values are uncalibrated model scores, "
                "not probabilities of disease or confirmed findings."
            )

            st.subheader("All 14 Model Scores")

            chart_df = df.set_index("finding")[["percentage"]]
            st.bar_chart(chart_df)

            with st.expander("View detailed prediction table"):
                st.dataframe(
                    df[["finding", "score", "percentage"]],
                    hide_index=True,
                    use_container_width=True
                )

        st.divider()
        st.subheader("AI-Generated Medical Information")

        st.markdown(
            result.get(
                "explanation",
                "No explanation was returned."
            )
        )

        st.subheader("Retrieved Medical Sources")

        sources = result.get("sources", [])

        if sources:
            for source in sources:
                st.write(f"• {source}")
        else:
            st.info("No source documents were returned.")

        st.divider()
        st.warning(
            result.get(
                "disclaimer",
                "Educational use only. Consult a qualified "
                "healthcare professional."
            )
        )

else:
    st.info(
        "Upload a de-identified chest X-ray image "
        "to begin the analysis."
    )

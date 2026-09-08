"""
Virtual Contrast MRI - Streamlit Application

Features:
- Upload T1, T2 and FLAIR MRI
- Generate Virtual T1CE
- Monte Carlo Dropout uncertainty map
- Confidence visualization
- Ground-truth evaluation
- Model information
"""

import sys
import os
import tempfile
import logging
from typing import Optional

# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

# ---------------------------------------------------------
# LIBRARIES
# ---------------------------------------------------------

import streamlit as st
import torch
import numpy as np
from PIL import Image
import nibabel as nib

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Virtual Contrast MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                #12304a 0%,
                #081521 45%,
                #050b12 100%
            );
        color: #e8f1f7;
    }

    .main {
        background: transparent;
    }

    /* =========================
       SIDEBAR
       ========================= */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0b1d2a 0%,
                #07131e 100%
            );
        border-right: 1px solid #1c3a4d;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #e8f7ff !important;
    }

    /* =========================
       HEADINGS
       ========================= */

    h1 {
        color: #e9fbff !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    h2, h3 {
        color: #bdefff !important;
    }

    /* =========================
       BUTTON
       ========================= */

    .stButton > button {
        width: 100%;
        background:
            linear-gradient(
                135deg,
                #00b8d9,
                #007c91
            );
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 20px;
        font-weight: 700;
        font-size: 16px;
        transition: 0.25s ease;
        box-shadow: 0 5px 18px rgba(0, 184, 217, 0.22);
    }

    .stButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #16d5f5,
                #009bb5
            );
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 184, 217, 0.35);
    }

    /* =========================
       FILE UPLOADER
       ========================= */

    [data-testid="stFileUploader"] {
        background: rgba(10, 31, 44, 0.65);
        border: 1px solid #1e465b;
        border-radius: 14px;
        padding: 8px;
    }

    /* =========================
       CARDS
       ========================= */

    .info-card {
        background:
            linear-gradient(
                145deg,
                rgba(16, 42, 57, 0.95),
                rgba(7, 22, 33, 0.95)
            );
        border: 1px solid #1d465a;
        border-radius: 16px;
        padding: 22px;
        margin: 10px 0;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.25);
    }

    .metric-card {
        background:
            linear-gradient(
                145deg,
                #102d3c,
                #0a1c28
            );
        border: 1px solid #1c5267;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }

    /* =========================
       STATUS
       ========================= */

    .status-success {
        background: rgba(0, 160, 120, 0.14);
        border: 1px solid #159a7b;
        color: #8ff5d5;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 10px 0;
    }

    .status-info {
        background: rgba(0, 184, 217, 0.10);
        border: 1px solid #177a91;
        color: #9defff;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 10px 0;
    }

    /* =========================
       TABS
       ========================= */

    button[data-baseweb="tab"] {
        color: #9bbbc8 !important;
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #48ddf7 !important;
    }

    /* =========================
       DIVIDER
       ========================= */

    hr {
        border-color: #214252 !important;
    }

    /* =========================
       CODE
       ========================= */

    code {
        color: #8deeff !important;
    }

    /* =========================
       FOOTER
       ========================= */

    .footer {
        text-align: center;
        color: #7193a1;
        padding: 25px;
        margin-top: 40px;
        border-top: 1px solid #1b3948;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# IMPORT PROJECT MODULES
# ---------------------------------------------------------

try:

    from models.attention_unet import VirtualContrastModel
    from predict import load_model, predict_single_slice
    from confidence_map import generate_confidence_map
    from metrics import evaluate_prediction

except ImportError as e:

    logger.error(f"Import error: {e}")

    st.error(
        f"Required project modules could not be imported: {e}"
    )

    st.stop()


# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

@st.cache_resource
def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_trained_model(checkpoint_path):

    device = get_device()

    try:

        model = load_model(
            checkpoint_path,
            device
        )

        return model, device

    except Exception as e:

        logger.error(
            f"Model loading error: {e}"
        )

        return None, device


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------

def preprocess_image(image: Image.Image):

    if image.mode != "L":
        image = image.convert("L")

    # IMPORTANT:
    # Training uses 128x128
    image = image.resize(
        (128, 128),
        Image.BILINEAR
    )

    img_array = np.array(
        image,
        dtype=np.float32
    )

    min_val = img_array.min()
    max_val = img_array.max()

    img_array = (
        img_array - min_val
    ) / (
        max_val - min_val + 1e-8
    )

    return img_array


# ---------------------------------------------------------
# NIFTI LOADER
# ---------------------------------------------------------

def load_nii_file(file) -> Optional[np.ndarray]:

    tmp_path = None

    try:

        suffix = ".nii.gz" if file.name.endswith(
            ".nii.gz"
        ) else ".nii"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(
                file.getvalue()
            )

            tmp_path = tmp.name

        img = nib.load(tmp_path)

        data = img.get_fdata()

        # Middle axial slice
        middle_slice = data[
            :,
            :,
            data.shape[2] // 2
        ]

        middle_slice = Image.fromarray(
            middle_slice.astype(np.float32)
        )

        # Convert to 128x128
        middle_slice = middle_slice.resize(
            (128, 128),
            Image.BILINEAR
        )

        middle_slice = np.array(
            middle_slice,
            dtype=np.float32
        )

        middle_slice = (
            middle_slice - middle_slice.min()
        ) / (
            middle_slice.max()
            - middle_slice.min()
            + 1e-8
        )

        return middle_slice

    except Exception as e:

        logger.error(
            f"NIfTI loading error: {e}"
        )

        return None

    finally:

        if tmp_path and os.path.exists(
            tmp_path
        ):
            os.unlink(tmp_path)


# ---------------------------------------------------------
# LOAD UPLOADED IMAGE
# ---------------------------------------------------------

def load_uploaded_image(file):

    if file is None:
        return None

    if file.name.lower().endswith(
        (".nii", ".nii.gz")
    ):

        return load_nii_file(file)

    return preprocess_image(
        Image.open(file)
    )


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

def show_header():

    st.title(
        "🧠 Virtual Contrast MRI"
    )

    st.markdown(
        """
        **AI-powered synthesis of T1CE MRI from T1, T2 and FLAIR**

        Generate a virtual contrast-enhanced MRI while
        visualizing model uncertainty.
        """
    )

    st.markdown("---")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    show_header()

    # =====================================================
    # SIDEBAR
    # =====================================================

    st.sidebar.title(
        "⚙️ Model Settings"
    )

    st.sidebar.markdown(
        "---"
    )

    checkpoint_path = st.sidebar.text_input(
        "Model Checkpoint",
        value="checkpoints/best_model.pth"
    )

    st.sidebar.markdown(
        "### 🔬 UNCERTAINTY"
    )

    n_samples = st.sidebar.slider(
        "MC Dropout Samples",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )

    st.sidebar.caption(
        "More samples = more stable uncertainty estimate, "
        "but slower inference."
    )

    uncertainty_threshold = st.sidebar.slider(
        "Uncertainty Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05
    )

    st.sidebar.caption(
        "Lower uncertainty → higher model confidence."
    )

    st.sidebar.markdown("---")

    st.sidebar.info(
        "GPU acceleration will be used automatically "
        "when CUDA is available."
    )

    # =====================================================
    # LOAD MODEL
    # =====================================================

    with st.spinner(
        "Loading trained model..."
    ):

        model, device = load_trained_model(
            checkpoint_path
        )

    if model is None:

        st.error(
            "❌ Model could not be loaded."
        )

        st.stop()

    st.markdown(
        f"""
        <div class="status-success">
        ✅ <b>Model Loaded Successfully</b>
        &nbsp;&nbsp; | &nbsp;&nbsp;
        Device: <b>{device}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "🔮 Prediction",
            "📦 Batch Processing",
            "📊 Model Information"
        ]
    )

    # =====================================================
    # TAB 1
    # =====================================================

    with tab1:

        st.header(
            "MRI Input"
        )

        col1, col2, col3 = st.columns(3)

        # -------------------------------------------------
        # T1
        # -------------------------------------------------

        with col1:

            st.subheader(
                "T1"
            )

            t1_file = st.file_uploader(
                "Upload T1",
                type=[
                    "png",
                    "jpg",
                    "jpeg",
                    "nii",
                    "gz"
                ],
                key="t1"
            )

            t1_img = None

            if t1_file:

                t1_img = load_uploaded_image(
                    t1_file
                )

                if t1_img is not None:

                    st.image(
                        t1_img,
                        caption="T1",
                        use_column_width=True
                    )

        # -------------------------------------------------
        # T2
        # -------------------------------------------------

        with col2:

            st.subheader(
                "T2"
            )

            t2_file = st.file_uploader(
                "Upload T2",
                type=[
                    "png",
                    "jpg",
                    "jpeg",
                    "nii",
                    "gz"
                ],
                key="t2"
            )

            t2_img = None

            if t2_file:

                t2_img = load_uploaded_image(
                    t2_file
                )

                if t2_img is not None:

                    st.image(
                        t2_img,
                        caption="T2",
                        use_column_width=True
                    )

        # -------------------------------------------------
        # FLAIR
        # -------------------------------------------------

        with col3:

            st.subheader(
                "FLAIR"
            )

            flair_file = st.file_uploader(
                "Upload FLAIR",
                type=[
                    "png",
                    "jpg",
                    "jpeg",
                    "nii",
                    "gz"
                ],
                key="flair"
            )

            flair_img = None

            if flair_file:

                flair_img = load_uploaded_image(
                    flair_file
                )

                if flair_img is not None:

                    st.image(
                        flair_img,
                        caption="FLAIR",
                        use_column_width=True
                    )

        st.markdown("---")

        # =================================================
        # PREDICT
        # =================================================

        if st.button(
            "🔮 Generate Virtual T1CE",
            type="primary"
        ):

            if (
                t1_img is None
                or t2_img is None
                or flair_img is None
            ):

                st.warning(
                    "Please upload T1, T2 and FLAIR images."
                )

                st.stop()

            with st.spinner(
                "Generating Virtual T1CE..."
            ):

                try:

                    # -------------------------------------
                    # MODEL INPUT
                    # -------------------------------------

                    x = torch.stack(
                        [
                            torch.from_numpy(
                                t1_img
                            ).float(),

                            torch.from_numpy(
                                t2_img
                            ).float(),

                            torch.from_numpy(
                                flair_img
                            ).float()
                        ],
                        dim=0
                    ).unsqueeze(0).to(device)

                    # -------------------------------------
                    # PREDICTION
                    # -------------------------------------

                    prediction = predict_single_slice(
                        model,
                        t1_img,
                        t2_img,
                        flair_img,
                        device
                    )

                    prediction = np.asarray(
                        prediction
                    ).squeeze()

                    # Normalize prediction to [0, 1]
                    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)

                    # -------------------------------------
                    # CONFIDENCE
                    # -------------------------------------

                    mean_pred, uncertainty = (
                        generate_confidence_map(
                            model,
                            x,
                            device,
                            n_samples=n_samples
                        )
                    )

                    confidence_map = uncertainty.squeeze()

                    # -------------------------------------
                    # DISPLAY
                    # -------------------------------------

                    st.success(
                        "✅ Virtual T1CE generated successfully!"
                    )

                    result1, result2 = st.columns(2)

                    with result1:

                        st.subheader(
                            "✨ Predicted T1CE"
                        )

                        st.image(
                            prediction,
                            caption="Virtual T1CE",
                            use_column_width=True
                        )

                    with result2:

                        st.subheader(
                            "🔬 Uncertainty Map"
                        )

                        st.image(
                            confidence_map,
                            caption=(
                                "MC Dropout Uncertainty"
                            ),
                            use_column_width=True
                        )

                    # -------------------------------------
                    # CONFIDENCE MASK
                    # -------------------------------------

                    thresholded_mask = (
                        confidence_map
                        < uncertainty_threshold
                    ).astype(
                        np.uint8
                    )

                    st.subheader(
                        "🎯 Confidence Mask"
                    )

                    st.image(
                        thresholded_mask,
                        caption=(
                            f"Uncertainty threshold: "
                            f"{uncertainty_threshold:.2f}"
                        ),
                        use_column_width=True
                    )

                    # -------------------------------------
                    # METRICS
                    # -------------------------------------

                    st.markdown("---")

                    st.subheader(
                        "📈 Evaluation"
                    )

                    gt_file = st.file_uploader(
                        "Upload Ground Truth T1CE (Optional)",
                        type=[
                            "png",
                            "jpg",
                            "jpeg",
                            "nii",
                            "gz"
                        ],
                        key="ground_truth"
                    )

                    if gt_file:

                        gt_img = load_uploaded_image(
                            gt_file
                        )

                        if gt_img is not None:

                            pred_tensor = (
                                torch.from_numpy(
                                    prediction
                                )
                                .float()
                                .unsqueeze(0)
                                .unsqueeze(0)
                            )

                            gt_tensor = (
                                torch.from_numpy(
                                    gt_img
                                )
                                .float()
                                .unsqueeze(0)
                                .unsqueeze(0)
                            )

                            metrics = (
                                evaluate_prediction(
                                    pred_tensor,
                                    gt_tensor
                                )
                            )

                            m1, m2, m3, m4 = (
                                st.columns(4)
                            )

                            with m1:

                                st.metric(
                                    "MAE",
                                    f"{metrics['MAE']:.4f}"
                                )

                            with m2:

                                st.metric(
                                    "MSE",
                                    f"{metrics['MSE']:.4f}"
                                )

                            with m3:

                                st.metric(
                                    "PSNR",
                                    f"{metrics['PSNR']:.2f} dB"
                                )

                            with m4:

                                st.metric(
                                    "SSIM",
                                    f"{metrics['SSIM']:.4f}"
                                )

                except Exception as e:

                    logger.exception(
                        "Prediction error"
                    )

                    st.error(
                        f"❌ Prediction error: {e}"
                    )

    # =====================================================
    # TAB 2
    # =====================================================

    with tab2:

        st.header(
            "📦 Batch Processing"
        )

        st.info(
            "Batch processing can be added here for "
            "processing multiple MRI patients automatically."
        )

        st.markdown(
            """
            ### Planned workflow

            1. Upload multiple patient folders
            2. Read T1, T2 and FLAIR
            3. Generate Virtual T1CE
            4. Generate uncertainty maps
            5. Save predictions
            6. Export results
            """
        )

    # =====================================================
    # TAB 3
    # =====================================================

    with tab3:

        st.header(
            "📊 Model Information"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                """
                <div class="info-card">

                <h3>🧠 Architecture</h3>

                <p>
                <b>Attention U-Net</b>
                </p>

                <p>
                Spatial Dimensions: 2D<br>
                Input Channels: 3<br>
                Output Channels: 1
                </p>

                <p>
                Channels:
                32 → 64 → 128 → 256 → 512
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                """
                <div class="info-card">

                <h3>🧬 MRI Modalities</h3>

                <p>
                <b>Input:</b><br>
                T1 + T2 + FLAIR
                </p>

                <p>
                <b>Target:</b><br>
                T1CE
                </p>

                <p>
                T1CE represents the
                contrast-enhanced MRI target.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader(
            "⚙️ Training Configuration"
        )

        st.markdown(
            """
            | Parameter | Value |
            |---|---|
            | Dataset | BraTS 2020 |
            | Model | Attention U-Net |
            | Input | T1 + T2 + FLAIR |
            | Target | T1CE |
            | Loss | L1 + SSIM + Edge |
            | Optimizer | AdamW |
            | Learning Rate | 1e-4 |
            | Batch Size | 16 |
            | Epochs | 50 |
            | Image Size | 128 × 128 |
            """
        )

        st.subheader(
            "🔬 Uncertainty Estimation"
        )

        st.markdown(
            """
            **Monte Carlo Dropout**

            Multiple forward passes are performed with
            dropout enabled. The variation between predictions
            is used as an uncertainty estimate.

            **Lower uncertainty → higher confidence**

            **Higher uncertainty → lower confidence**
            """
        )

        st.subheader(
            "📈 Evaluation Metrics"
        )

        metric1, metric2, metric3, metric4 = (
            st.columns(4)
        )

        with metric1:
            st.metric(
                "MAE",
                "Lower is better"
            )

        with metric2:
            st.metric(
                "MSE",
                "Lower is better"
            )

        with metric3:
            st.metric(
                "PSNR",
                "Higher is better"
            )

        with metric4:
            st.metric(
                "SSIM",
                "Higher is better"
            )

    # =====================================================
    # FOOTER
    # =====================================================

    st.markdown(
        """
        <div class="footer">

        🧠 <b>Virtual Contrast MRI</b><br>

        AI-based Medical Image Synthesis & Uncertainty Quantification

        <br><br>

        Built with PyTorch • MONAI • Attention U-Net • Streamlit

        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
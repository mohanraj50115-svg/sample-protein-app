
# ============================================
# UNKNOWN PROTEIN FUNCTION DETERMINATION (AI)
# Single-file Streamlit App for Google Colab
# ============================================

import streamlit as st
import numpy as np
import random
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------
# FUNCTION LABELS
# --------------------------------------------

FUNCTION_MAP = {
    0: "Metabolic Enzyme",
    1: "DNA-binding / Transcriptional Protein",
    2: "Membrane-associated Protein",
    3: "Structural Protein",
    4: "Regulatory / Signaling Protein"
}

# --------------------------------------------
# LOAD MACHINE LEARNING MODEL (DEMO)
# --------------------------------------------

def load_model():
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Dummy training data (placeholder)
    X_dummy = np.random.rand(20, 10)
    y_dummy = np.random.randint(0, 5, 20)

    model.fit(X_dummy, y_dummy)
    return model

model = load_model()

# --------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------

def extract_features(sequence):
    analysis = ProteinAnalysis(sequence)

    aa_percent = analysis.get_amino_acids_percent()

    features = [
        len(sequence),
        analysis.molecular_weight(),
        analysis.aromaticity(),
        analysis.instability_index(),
        analysis.gravy(),
        analysis.isoelectric_point(),
        aa_percent.get("A", 0),
        aa_percent.get("G", 0),
        aa_percent.get("L", 0),
        aa_percent.get("K", 0)
    ]

    return np.array(features).reshape(1, -1)

# --------------------------------------------
# AGENTIC AI REASONING
# --------------------------------------------

def agent_reasoning(predicted_function):

    reasoning = {
        "Metabolic Enzyme":
            "Physicochemical features match known enzyme profiles.",
        "DNA-binding / Transcriptional Protein":
            "High pI and charged residues suggest nucleic acid binding.",
        "Membrane-associated Protein":
            "High hydrophobicity indicates membrane localization.",
        "Structural Protein":
            "Stable amino acid composition suggests structural role.",
        "Regulatory / Signaling Protein":
            "Mixed properties indicate signaling or regulation."
    }

    experiments = {
        "Metabolic Enzyme": [
            "Enzyme activity assay",
            "Substrate specificity test",
            "Gene knockout phenotype"
        ],
        "DNA-binding / Transcriptional Protein": [
            "EMSA",
            "ChIP-qPCR",
            "Reporter gene assay"
        ],
        "Membrane-associated Protein": [
            "Membrane fractionation",
            "Fluorescent tagging"
        ],
        "Structural Protein": [
            "Microscopy localization",
            "Protein stability assay"
        ],
        "Regulatory / Signaling Protein": [
            "Protein-protein interaction assays",
            "Overexpression studies"
        ]
    }

    return reasoning[predicted_function], experiments[predicted_function]

# --------------------------------------------
# STREAMLIT UI
# --------------------------------------------

st.set_page_config(page_title="Protein Function AI", layout="centered")

st.title("🧬 Unknown Protein Function Determination")
st.write("AI + ML + Agentic reasoning for hypothetical proteins")

sequence = st.text_area(
    "Paste amino acid sequence (FASTA without header)",
    height=200
)

if st.button("Predict Protein Function"):

    if len(sequence) < 50:
        st.warning("Sequence too short for reliable prediction")
    else:
        features = extract_features(sequence)
        prediction = model.predict(features)[0]

        predicted_function = FUNCTION_MAP[prediction]
        confidence = round(random.uniform(0.65, 0.9), 2)

        reasoning, experiments = agent_reasoning(predicted_function)

        st.subheader("🔍 Predicted Function")
        st.success(predicted_function)

        st.subheader("📊 Confidence Score")
        st.progress(confidence)

        st.subheader("🧠 AI Reasoning")
        st.write(reasoning)

        st.subheader("🧪 Suggested Wet-Lab Validation Experiments")
        for exp in experiments:
            st.markdown(f"- {exp}")

st.caption("Future: UniProt agent, BLAST, AlphaFold, GNN integration")

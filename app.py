import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
import joblib

# Load model and features
model = joblib.load("rf_model.joblib")
features = joblib.load("features.joblib")

# Descriptor function
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.NOCount(mol)
    ]
    return pd.DataFrame([desc], columns=features)

# App UI
st.title("🧪 AlzPredictor - Drug Solubility (logS) & Activity")

smiles = st.text_input("Enter Drug SMILES")

if st.button("Predict"):
    desc = compute_descriptors(smiles)
    if desc is None:
        st.error("❌ Invalid SMILES string.")
    else:
        prediction = model.predict(desc)[0]
        st.success(f"Predicted logS: {prediction:.2f}")
        if prediction > 6:
            st.info("🧬 Drug is **Active** against Alzheimer's")
        elif prediction < 5:
            st.warning("⚠️ Drug is **Inactive**")
        else:
            st.info("🔬 Drug is **Moderately Active**")

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import pickle

# Load the trained model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load feature names if you saved them
with open('features.pkl', 'rb') as file:
    feature_names = pickle.load(file)

# Function to calculate molecular descriptors
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptor_functions = {
        'MolWt': Descriptors.MolWt,
        'MolLogP': Descriptors.MolLogP,
        'NumHDonors': Descriptors.NumHDonors,
        'NumHAcceptors': Descriptors.NumHAcceptors,
        'TPSA': Descriptors.TPSA,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'RingCount': Descriptors.RingCount,
    }
    
    descriptors = {name: func(mol) for name, func in descriptor_functions.items()}
    
    return pd.DataFrame([descriptors])

# Streamlit UI
st.title("Log S Prediction and Drug Activity Classifier")
st.write("Enter the SMILES of a compound to predict its solubility (log S) and check if it's active.")

smiles_input = st.text_input("Enter SMILES:")

if st.button("Predict"):
    if not smiles_input:
        st.warning("Please enter a valid SMILES string.")
    else:
        descriptors_df = calculate_descriptors(smiles_input)
        if descriptors_df is None:
            st.error("Invalid SMILES string.")
        else:
            # Ensure all required features are present
            for col in feature_names:
                if col not in descriptors_df.columns:
                    descriptors_df[col] = 0
            descriptors_df = descriptors_df[feature_names]
            
            # Predict log S
            log_s = model.predict(descriptors_df)[0]
            st.success(f"Predicted log S: {log_s:.2f}")

            # Drug activity based on log S
            if log_s > -4:
                st.markdown("### 🟢 The drug is **Active**")
            else:
                st.markdown("### 🔴 The drug is **Inactive**")

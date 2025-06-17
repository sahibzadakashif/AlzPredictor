import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

# Load your trained RF model
model = joblib.load("rf_model.pkl")

descriptor_columns = [
    'apol', 'arorings', 'ast_fraglike', 'ast_fraglike_ext', 'ast_violation',
    'ast_violation_ext', 'a_acc', 'a_acid', 'a_aro', 'a_base', 'a_count',
    'a_don', 'a_donacc', 'a_heavy', 'a_hyd', 'a_IC', 'a_ICM', 'a_nB',
    'a_nBr', 'a_nC', 'a_nCl', 'a_nF', 'a_nH', 'a_nI', 'a_nN', 'a_nNO',
    'a_nO', 'a_nP', 'a_nS', 'balabanJ', 'BCUT_PEOE_0', 'BCUT_PEOE_1',
    'BCUT_PEOE_2', 'BCUT_PEOE_3', 'BCUT_SLOGP_0', 'BCUT_SLOGP_1',
    'BCUT_SLOGP_2', 'BCUT_SLOGP_3', 'BCUT_SMR_0', 'BCUT_SMR_1',
    'BCUT_SMR_2', 'BCUT_SMR_3', 'bpol', 'b_1rotN', 'b_1rotR', 'b_ar',
    'b_count', 'b_double', 'b_heavy', 'b_max1len', 'b_rotN', 'b_rotR',
    'b_single', 'b_triple', 'chi0', 'chi0v', 'chi0v_C', 'chi0_C', 'chi1',
    'chi1v', 'chi1v_C', 'chi1_C', 'chiral', 'chiral_u', 'density',
    'diameter', 'FCharge', 'GCUT_PEOE_0', 'GCUT_PEOE_1', 'GCUT_PEOE_2',
    'GCUT_PEOE_3', 'GCUT_SLOGP_0', 'GCUT_SLOGP_1', 'GCUT_SLOGP_2',
    'GCUT_SLOGP_3', 'GCUT_SMR_0', 'GCUT_SMR_1', 'GCUT_SMR_2', 'GCUT_SMR_3',
    'h_ema', 'h_emd', 'h_emd_C', 'h_logD', 'h_logP', 'h_logS', 'h_log_dbo',
    'h_log_pbo', 'h_mr', 'h_pavgQ', 'h_pKa', 'h_pKb', 'h_pstates',
    'h_pstrain', 'Kier1', 'Kier2', 'Kier3', 'KierA1', 'KierA2', 'KierA3',
    'KierFlex', 'lip_acc', 'lip_don', 'logP(o/w)', 'mr', 'n_atom',
    'n_atom_nH', 'n_ring', 'nAromRing', 'nBase', 'nBonds', 'nCIC',
    'nCount', 'nDon', 'nHBAcc', 'nHet', 'nHetAromRing', 'nRot',
    'nRot_b', 'nStereo', 'nT3', 'nT4', 'nX', 'PEOE_VSA+0', 'PEOE_VSA+1',
    'PEOE_VSA+2', 'PEOE_VSA+3', 'PEOE_VSA-1', 'PEOE_VSA-2', 'PEOE_VSA-3',
    'PEOE_VSA_FHYD', 'PEOE_VSA_POL', 'PEOE_VSA_POS', 'PEOE_VSA_PPOS',
    'Q_VSA_FHYD', 'Q_VSA_FNEG', 'Q_VSA_FPOL', 'Q_VSA_FPOS', 'Q_VSA_POL',
    'Q_VSA_POS', 'reactive', 'ring_atom', 'ring_n4', 'rotatable_bond',
    'SlogP', 'SlogP_VSA+0', 'SlogP_VSA+1', 'SlogP_VSA+2', 'SlogP_VSA+3',
    'SlogP_VSA-1', 'SlogP_VSA-2', 'SlogP_VSA-3', 'SlogP_VSA4', 'SlogP_VSA9',
    'SlogP_VSA_FHYD', 'SlogP_VSA_POL', 'SlogP_VSA_POS', 'SlogP_VSA_PPOS',
    'SMR_VSA+0', 'SMR_VSA+1', 'SMR_VSA+2', 'SMR_VSA+3', 'SMR_VSA+4',
    'SMR_VSA-1', 'SMR_VSA-2', 'SMR_VSA-3', 'SMR_VSA4', 'SMR_VSA5',
    'SMR_VSA6', 'SMR_VSA_FHYD', 'SMR_VSA_POL', 'SMR_VSA_POS',
    'SMR_VSA_PPOS', 'surface_area', 'TPSA', 'vabc', 'vsa_acc',
    'vsa_don', 'vsa_other', 'vsa_pol'
]


# Simulated descriptor calculator (in practice, MOE descriptors should be precomputed or matched)
def calculate_rdkit_approx_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    # Manually extract a subset of RDKit-approximated values
    descriptors = {
        'apol': Descriptors.TPSA(mol),
        'a_heavy': Descriptors.HeavyAtomCount(mol),
        'a_don': Descriptors.NumHDonors(mol),
        'a_acc': Descriptors.NumHAcceptors(mol),
        'a_aro': Chem.rdMolDescriptors.CalcNumAromaticRings(mol),
        'logP(o/w)': Descriptors.MolLogP(mol),
        'mr': Descriptors.MolMR(mol),
        'Weight': Descriptors.MolWt(mol),
        'TPSA': Descriptors.TPSA(mol),
        'zagreb': Descriptors.ZagrebIndex(mol),
        # Fill other values with dummy or mean if not computable
    }

    for col in descriptor_columns:
        if col not in descriptors:
            descriptors[col] = 0.0  # fallback default

    return pd.DataFrame([descriptors])

# Streamlit UI
st.title("🔬 LogS Predictor (Solubility) from SMILES")
st.write("Predict aqueous solubility (LogS) using a Random Forest model trained on MOE descriptors.")

# SMILES input
smiles_input = st.text_input("🧪 Enter SMILES string:")

if smiles_input:
    desc_df = calculate_rdkit_approx_descriptors(smiles_input)

    if desc_df is None:
        st.error("❌ Invalid SMILES. Please check your input.")
    else:
        try:
            prediction = model.predict(desc_df[descriptor_columns])
            st.success(f"✅ Predicted LogS: **{prediction[0]:.4f}**")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

            if log_s > -4:
                st.markdown("### 🟢 The drug is **Active**")
            else:
                st.markdown("### 🔴 The drug is **Inactive**")

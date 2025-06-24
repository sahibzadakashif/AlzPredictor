import streamlit as st
import pandas as pd
import numpy as np
import base64
from rdkit import Chem
from rdkit.Chem import Descriptors

# Streamlit Page Setup
st.set_page_config(page_title="AlzPredictor", layout="wide")

# Function to compute TPSA
def calculate_tpsa(smiles_list):
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            tpsa = Descriptors.TPSA(mol)
            results.append((smi, round(tpsa, 2)))
        else:
            results.append((smi, None))
    return pd.DataFrame(results, columns=["SMILES", "TPSA (Å²)"])

# Download link for CSV
def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="TPSA_predictions.csv">📥 Download CSV</a>'

# Main App
def main():
    st.title("🔬 AlzPredictor")
    st.markdown("Calculate the **Topological Polar Surface Area (TPSA)** from Canonical SMILES of Drugs targeting significant biomarker genes in Alzheimer's Disease.")

    input_method = st.radio("Choose Input Method", ["Paste SMILES", "Upload File"])

    if input_method == "Paste SMILES":
        smiles_input = st.text_area("Enter SMILES strings (one per line)")
        if st.button("Predict TPSA"):
            smiles_list = [s.strip() for s in smiles_input.splitlines() if s.strip()]
            if not smiles_list:
                st.warning("Please enter valid SMILES.")
            else:
                df = calculate_tpsa(smiles_list)
                st.success("✅ TPSA prediction complete!")
                st.dataframe(df)
                st.markdown(get_download_link(df), unsafe_allow_html=True)

    else:
        file = st.file_uploader("Upload a CSV or TXT file with SMILES", type=["csv", "txt"])
        if file and st.button("Predict TPSA"):
            try:
                df = pd.read_csv(file, header=None)
                smiles_list = df.iloc[:, 0].dropna().astype(str).tolist()
                results = calculate_tpsa(smiles_list)
                st.success("✅ TPSA prediction complete!")
                st.dataframe(results)
                st.markdown(get_download_link(results), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.markdown("---")
    st.markdown("## AlzPredictor Contributors:")

    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
            <div style='line-height: 1.3; color: #000000;'>
                <h3 style='color:#800000;'>Dr. Kashif Iqbal Sahibzada</h3>
                Assistant Professor<br>
                Department of Health Professional Technologies,<br>
                Faculty of Allied Health Sciences,<br>
                The University of Lahore<br>
                Post-Doctoral Fellow<br>
                Henan University of Technology, Zhengzhou, China<br>
                <b>Email:</b> kashif.iqbal@dhpt.uol.edu.pk | kashif.iqbal@haut.edu.cn
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='line-height: 1.3; color: #000000;'>
                <h3 style='color:#800000;'>Dr. Andleeb Batool</h3>
                Assistant Professor<br>
                Department of Zoology<br>
                Government College University, Lahore<br>
                <b>Email:</b> andleeb.batool@gcu.edu.pk
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style='line-height: 1.3; color: #000000;'>
                <h3 style='color:#800000;'>Shumaila Shahid</h3>
                MS Biochemistry<br>
                School of Biochemistry and Biotechnology<br>
                University of the Punjab, Lahore<br>
                <b>Email:</b> shumaila.ms.sbb@pu.edu.pk
            </div>
        """, unsafe_allow_html=True)

    

if __name__ == "__main__":
    main()

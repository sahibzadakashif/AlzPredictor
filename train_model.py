import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv("/content/final_output_file.csv")  # Must have 'canonical_smiles' and 'Topological_Polar_Surface_Area'
smiles_list = data["canonical_smiles"]
targets = data["Topological_Polar_Surface_Area"]

# Convert SMILES to Morgan fingerprints
def smiles_to_fp(smiles, radius=2, nBits=1024):  # ✅ Reduced size from 2048 → 1024
    features = []
    valid_targets = []
    for smi, target in zip(smiles, targets):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            features.append(np.array(fp))
            valid_targets.append(target if np.isfinite(target) else np.nan)
    return np.array(features), np.array(valid_targets)

# Generate feature matrix and clean
X, y = smiles_to_fp(smiles_list)
X = X[~np.isnan(y)]
y = y[~np.isnan(y)]

# Train model with reduced complexity
model = RandomForestRegressor(
    n_estimators=50,      # ✅ Reduced number of trees
    max_depth=15,         # ✅ Limit depth to reduce memory footprint
    random_state=42,
    n_jobs=-1             # Uses all available cores (optional)
)
model.fit(X, y)

# Save the model with compression
joblib.dump(model, "model.pkl", compress=3)  # ✅ Compression level 3
print("✅ Model trained and saved as compressed model.pkl")

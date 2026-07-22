import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

df = pd.read_csv("../data/gyrB_clean.csv")


def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {
        "MolecularWeight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "RingCount": Descriptors.RingCount(mol),
        "HeavyAtoms": Descriptors.HeavyAtomCount(mol),
    }

descriptor_list = []

for smiles in df["canonical_smiles"]:
    descriptor_list.append(calculate_descriptors(smiles))

descriptor_df = pd.DataFrame(descriptor_list)


final_df = pd.concat(
    [
        df.reset_index(drop=True),
        descriptor_df
    ],
    axis=1
)

# print(final_df.head())

# print(final_df.columns)

final_df.to_csv("../data/gyrB_descriptors.csv", index=False)

print("Descriptor dataset saved successfully!")
print("Shape:", final_df.shape)
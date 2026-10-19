import unittest
import pandas as pd
from clean_data import clean as clean_internal
from clean_external_data import clean as clean_external
from pic50_conversion import convert
from download_external_data import fetch_activities
from feature_engineering import build_tables

class FakeResponse:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): pass
    def json(self): return self.payload
class FakeSession:
    def __init__(self, pages): self.pages=iter(pages); self.calls=[]
    def get(self, url, params, timeout): self.calls.append((params.copy(),timeout)); return FakeResponse(next(self.pages))

class LabelPolicyTests(unittest.TestCase):
    def test_internal_exact_only_and_provenance(self):
        base = dict(activity_id=[1,2,3,4,5], molecule_chembl_id=["a","b","c","d","e"],
                    canonical_smiles=["C"]*5, standard_type=["IC50"]*5,
                    standard_relation=["=","<",">","=","="], standard_value=[10,20,30,0,-2],
                    standard_units=["nM"]*5, data_validity_comment=[None]*5, assay_chembl_id=["assay"]*5)
        out, counts = clean_internal(pd.DataFrame(base))
        self.assertEqual(out["activity_id"].tolist(), [1])
        self.assertIn("standard_relation", out.columns); self.assertIn("assay_chembl_id", out.columns)
        self.assertEqual(counts["retained_rows"], 1)
    def test_external_keeps_provenance_and_rejects_censored_or_flagged(self):
        d = pd.DataFrame({"Molecule ChEMBL ID":["a","b","c"],"Smiles":["C"]*3,
            "Standard Type":["IC50"]*3,"Standard Relation":["=","<","="],
            "Standard Value":[10,20,30],"Standard Units":["nM"]*3,
            "Data Validity Comment":[None,None,"Outside typical range"],"Assay ChEMBL ID":["x"]*3})
        out, counts = clean_external(d)
        self.assertEqual(out["molecule_chembl_id"].tolist(), ["a"])
        self.assertIn("standard_relation", out.columns); self.assertEqual(counts["retained_rows"], 1)
    def test_pic50_rejects_nonpositive_and_non_nM(self):
        good = convert(pd.DataFrame({"standard_value":[10],"standard_units":["nM"]}))
        self.assertAlmostEqual(good.pIC50.iloc[0], 8.0)
        with self.assertRaises(ValueError): convert(pd.DataFrame({"standard_value":[0],"standard_units":["nM"]}))
    def test_downloader_paginates_and_preserves_all_rows(self):
        session=FakeSession([{"activities":[{"activity_id":1},{"activity_id":2}],"page_meta":{"total_count":3}},
                             {"activities":[{"activity_id":3}],"page_meta":{"total_count":3}}])
        rows,pages=fetch_activities(session,target_id="T",page_size=2)
        self.assertEqual([r["activity_id"] for r in rows],[1,2,3]); self.assertEqual(pages,2)
        self.assertEqual([c[0]["offset"] for c in session.calls],[0,2]); self.assertTrue(all(c[1] for c in session.calls))
    def test_model_table_excludes_metadata_and_sidecar_preserves_identity(self):
        row = {name: [1.0] for name in ["MolecularWeight", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "RingCount", "HeavyAtoms", "standard_value"]}
        row.update({"activity_id": [7], "molecule_chembl_id": ["CHEMBL1"], "standard_relation": ["="], "standard_units": ["nM"]})
        model, provenance = build_tables(pd.DataFrame(row))
        self.assertNotIn("molecule_chembl_id", model.columns)
        self.assertEqual(provenance.source_row.tolist(), [0])
        self.assertEqual(provenance.molecule_chembl_id.tolist(), ["CHEMBL1"])
    def test_downloader_fails_on_early_empty_page(self):
        session=FakeSession([{"activities":[],"page_meta":{"total_count":1}}])
        with self.assertRaisesRegex(RuntimeError,"empty page"):
            fetch_activities(session,target_id="T",page_size=2)

if __name__ == "__main__": unittest.main()

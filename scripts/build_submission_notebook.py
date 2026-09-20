"""Synchronize the final submission cells with their tested companion scripts."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/building-transformers-v2.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))
# Re-running this generator replaces only its own final section.
for index, cell in enumerate(notebook["cells"]):
    if "".join(cell["source"]).startswith("## 7. Final Submission"):
        notebook["cells"] = notebook["cells"][:index]
        break
text = """## 7. Final Submission

Generate `submission.csv` with the verified Transformer checkpoint, then submit
it as a late entry to IEEE-CIS. The competition evaluates ROC-AUC.

For inference only, attach this notebook's version 1 output and run the core imports,
dataset discovery, GPU check, FTTransformer class definition, and the two cells below.
For a fresh end-to-end experiment, run the whole notebook. Pretrained benchmarks
are separate from the submission model.

The final CSV contains every test TransactionID once, in sample-submission order.
Inference uses batches of 64 and never fits preprocessing on test data.
`results.json` keeps the main validation and small foundation-model comparison separate.
The Kaggle score is recorded only after the site evaluates the submission.
"""
notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
for name in ("kaggle_submission.py", "kaggle_finalize_cell.py"):
    code = (root / "scripts" / name).read_text()
    ast.parse(code)
    notebook["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None,
                              "outputs": [], "source": code.splitlines(keepends=True)})
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]))
path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print("Validated notebook cells:", len(notebook["cells"]))

# === Restore the verified Transformer and generate the Kaggle submission ===
# Run core imports, dataset discovery, GPU check, and the FTTransformer class cell first.
import shutil
import joblib

# Prefer the current run; otherwise restore this notebook's attached saved output.
if not (ARTIFACT_DIR / "preprocessing.pkl").exists():
    candidates = [path.parent for path in Path("/kaggle/input").rglob("preprocessing.pkl")
                  if "notebook70df114be2" in str(path) and (path.parent / "model.pt").exists()]
    if len(candidates) != 1:
        raise FileNotFoundError("Attach version 1 output of notebook70df114be2, or run the training cells first.")
    for path in candidates[0].iterdir():
        if path.is_file():
            shutil.copy2(path, ARTIFACT_DIR / path.name)
    print("Restored artifacts from:", candidates[0])

checkpoint = torch.load(ARTIFACT_DIR / "model.pt", map_location="cpu", weights_only=True)
preprocessing = joblib.load(ARTIFACT_DIR / "preprocessing.pkl")
submission_model = FTTransformer(
    checkpoint["num_numeric_features"], checkpoint["category_cardinalities"],
    checkpoint["d_model"], checkpoint["nhead"], checkpoint["num_layers"], checkpoint["dropout"],
)
submission_model.load_state_dict(checkpoint["model_state"])
submission = generate_submission(submission_model, preprocessing, DATA_DIR,
                                 "/kaggle/working", device=DEVICE, batch_size=64)

# Bind the CSV to the exact checkpoint used; do not invent a Kaggle score.
manifest_path = Path("/kaggle/working/submission_manifest.json")
manifest = json.loads(manifest_path.read_text())
manifest["checkpoint_sha256"] = hashlib.sha256((ARTIFACT_DIR / "model.pt").read_bytes()).hexdigest()
manifest["validation_metrics"] = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
manifest_path.write_text(json.dumps(manifest, indent=2))

# Collate results into distinct evaluation groups.
main_results = [json.loads((ARTIFACT_DIR / name).read_text())
                for name in ("lightgbm_metrics.json", "metrics.json")]
foundation_path = ARTIFACT_DIR / "foundation_benchmarks.json"
foundation_results = json.loads(foundation_path.read_text()) if foundation_path.exists() else []
results = {"main_validation": main_results, "small_matched_benchmark": foundation_results,
           "kaggle_submission": {"file": "submission.csv", "status": "ready", "score": None}}
Path("/kaggle/working/results.json").write_text(json.dumps(results, indent=2))
print("Main validation: 40,000 training / 10,000 validation rows, 432 features")
display(pd.DataFrame(main_results).drop(columns=["history"], errors="ignore"))
print("Small matched benchmark: 2,000 training / 1,000 validation rows, 100 features")
display(pd.DataFrame(foundation_results))
print("Validated submission.csv is ready. Kaggle scoring is a separate step.")

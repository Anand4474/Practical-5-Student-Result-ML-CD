# Practical 5: Continuous Delivery of an ML Prediction Application

This project extends the Student Result Prediction CI workflow with a Flask prediction API and a GitHub Actions release-candidate artifact.

## Workflow
1. Install Python dependencies.
2. Train the model and generate `student_result_model.pkl`, `metrics.json`, and `student_results.csv`.
3. Check the ML quality gate (minimum accuracy 0.80).
4. Run ML pipeline tests.
5. Run prediction API tests.
6. Package the application as `release_candidate/`.
7. Upload the `student-result-ml-release` artifact.

## Run locally (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python quality_gate.py
python -m unittest test_ml_pipeline.py -v
python -m unittest test_app.py -v
python app.py
```

Open `http://127.0.0.1:5000/` for the health check.

## Prediction request
Send a POST request to `/predict` with JSON:
```json
{
  "attendance": 90,
  "internal_marks": 85,
  "assignment_marks": 88,
  "previous_score": 80
}
```

## Download the release
After a successful GitHub Actions run, open **Actions**, select the run, and download `student-result-ml-release` from the Artifacts section. Extract it and install dependencies with `pip install -r requirements.txt`, then run `python app.py`.

This practical prepares a release candidate; it does not deploy to production.

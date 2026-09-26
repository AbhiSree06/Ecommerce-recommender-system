# E-Commerce Recommender

This project contains the folder structure for an e-commerce recommender system using ML-based embeddings and sequential pattern mining.

## Structure

- `data/raw/` - raw JSONL inputs
- `data/processed/` - cleaned/processed datasets
- `data/synthetic/` - synthetic data generation outputs
- `ml/preprocessing/` - data cleaning and feature engineering
- `ml/embeddings/` - embedding generation and index creation
- `ml/sequential_mining/` - sequential recommendation logic
- `ml/evaluation/` - metrics and evaluation scripts
- `backend/app/` - FastAPI application
- `backend/recommender/` - recommendation service logic
- `frontend/` - frontend app files
- `models/` - trained model artifacts
- `scripts/` - project automation scripts
- `tests/` - unit and integration tests

## Getting started

1. Create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your raw data JSONL files to `data/raw/`.
4. Start implementing preprocessing, embedding generation, and recommender services.

## Notes

This is the initial scaffold only. The project structure is ready for further ML and backend development.

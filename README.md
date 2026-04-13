# rag-langgraph

## RAG observability

This project now includes:

- MLflow run logging for index build metrics.
- MLflow tracing for each interactive chat request.
- DeepEval-based offline RAG evaluation with results logged back to MLflow.

## Environment variables

The existing LLM configuration is reused for online chat:

- `LLM_API_KEY`
- `LLM_MODEL_ID`
- `LLM_BASE_URL`

Optional MLflow settings:

- `MLFLOW_ENABLED=true`
- `MLFLOW_TRACKING_URI=file:./mlruns`
- `MLFLOW_EXPERIMENT_NAME=study-python-rag`
- `MLFLOW_TRACE_REQUESTS=true`

Optional DeepEval settings:

- `DEEPEVAL_MODEL_ID`
- `DEEPEVAL_API_KEY`
- `DEEPEVAL_DATASET_PATH=./docs/rag_eval_dataset.example.json`
- `DEEPEVAL_THRESHOLD=0.5`

If `DEEPEVAL_MODEL_ID` or `DEEPEVAL_API_KEY` is not set, the CLI falls back to
`LLM_MODEL_ID` and `LLM_API_KEY`.

## Usage

Start the interactive chat CLI:

```bash
python main.py chat
```

Run DeepEval on a dataset:

```bash
python main.py evaluate --dataset docs/rag_eval_dataset.example.json
```

Open the local MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

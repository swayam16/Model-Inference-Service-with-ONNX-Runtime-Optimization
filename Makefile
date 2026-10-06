.PHONY: install model benchmark serve test lint load docker

install:        ## Install everything needed to export, benchmark and test
	pip install -r requirements-dev.txt

model:          ## Download the model and export it to ONNX (FP32 + INT8)
	python -m scripts.export_onnx

benchmark:      ## PyTorch vs ONNX Runtime latency on this machine
	python -m scripts.benchmark

serve:          ## Run the API and dashboard on http://localhost:8000
	uvicorn app.main:app --port 8000

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

load:           ## Send load to a running service
	python -m scripts.load_test --url http://localhost:8000 --requests 1500 --concurrency 32

docker:         ## Build and run the container on http://localhost:8000
	docker compose up --build

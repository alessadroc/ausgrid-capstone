.PHONY: inspect preprocess test clean

inspect:
	python scripts/inspect_data.py --input data/raw

preprocess:
	python scripts/run_preprocessing.py --input data/raw --output data/processed \
		--start-date 2023-01-01 --end-date 2025-12-31 --stream

test:
	pytest tests/ -v

clean:
	rm -rf data/processed/* results/metrics/*
	find . -name __pycache__ -type d -exec rm -rf {} +

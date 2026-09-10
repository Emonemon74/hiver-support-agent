# Hiver Support Agent — dev commands

PY=.venv/bin/python

.PHONY: install data profile dataset intents golden pipeline eval report test

install:
	uv pip install -r requirements.txt

data:
	$(PY) -m src.get_data

profile:
	$(PY) -m src.profile_brands

dataset:
	$(PY) -m src.build_dataset

eval:
	$(PY) -m src.eval.run_all

test:
	.venv/bin/pytest -q

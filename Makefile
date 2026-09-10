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

agent:
	$(PY) -m src.eval.run_agent subset

routing:
	$(PY) -m src.eval.rerun_routing

judge-agreement:
	$(PY) -m src.eval.judge_validation

eval:
	$(PY) -m src.eval.run_all

test:
	.venv/bin/pytest -q

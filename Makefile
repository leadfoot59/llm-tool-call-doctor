PYTHON ?= python3
UPSTREAM ?= http://127.0.0.1:11434/v1
MODEL ?= qwen2.5-coder:7b
PORT ?= 11500
FAKE_PORT ?= 11888

.PHONY: help test probe shim score agent case-files fake score-fake clean

help:
	@echo "make test                       run the full test suite"
	@echo "make probe  MODEL=<id>          identify the dialect a model emits"
	@echo "make shim   UPSTREAM=<url>      run the repair shim on :$(PORT)"
	@echo "make score  MODEL=<id>          score raw backend vs shim"
	@echo "make agent  MODEL=<id>          run agent tasks across three tool menus"
	@echo "make fake                       run the misbehaving test backend on :$(FAKE_PORT)"
	@echo "make score-fake                 score every fake dialect through the shim"
	@echo "make clean                      remove caches and scratch output"

test:
	$(PYTHON) tests/run_all.py

probe:
	$(PYTHON) -m toolcalldoctor probe --base-url $(UPSTREAM) --model $(MODEL)

shim:
	$(PYTHON) -m toolcalldoctor shim --upstream $(UPSTREAM) --port $(PORT) --trace shim-trace.jsonl

score:
	$(PYTHON) -m toolcalldoctor score --cases cases/basic.json --model $(MODEL) \
		--base-url http://127.0.0.1:$(PORT)/v1 --baseline-url $(UPSTREAM)

# Answers "do I need to trim the tool menu?" with data. Executes model-written
# shell commands in a throwaway directory -- see the safety note in the README.
agent:
	$(PYTHON) -m toolcalldoctor agent --model $(MODEL) --base-url $(UPSTREAM) \
		--menus all "bash read edit write glob grep list_dir" bash

fake:
	$(PYTHON) tests/fake_backend.py --port $(FAKE_PORT)

score-fake:
	@for d in hermes llama3 mistral qwen gemma glm minimax python fenced single_key truncated; do \
		echo "=== fake/$$d ==="; \
		$(PYTHON) -m toolcalldoctor score --cases tests/data/fake_cases.json --model fake/$$d \
			--base-url http://127.0.0.1:$(PORT)/v1 \
			--baseline-url http://127.0.0.1:$(FAKE_PORT)/v1 | tail -12; \
	done

clean:
	rm -rf .tmp __pycache__ */__pycache__ *.egg-info build dist
	find . -name '*.pyc' -delete

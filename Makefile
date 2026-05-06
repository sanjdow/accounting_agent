.PHONY: help setup install run ollama-pull clean reset

help:
	@echo "Targets:"
	@echo "  setup       - create .venv and install dependencies"
	@echo "  install     - install dependencies into current env"
	@echo "  run         - start Streamlit UI on :8501"
	@echo "  ollama-pull - pull the default Ollama model"
	@echo "  clean       - remove generated DBs and reports"
	@echo "  reset       - clean + re-create .venv"

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo ""
	@echo "Next: source .venv/bin/activate && make run"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

run:
	streamlit run app.py

ollama-pull:
	ollama pull llama3.1:8b

clean:
	rm -rf close.db chroma_db reports_out/*.pdf reports_out/*.xlsx
	@echo "Cleaned DBs and reports."

reset: clean
	rm -rf .venv
	$(MAKE) setup

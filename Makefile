.PHONY: install run clean

VENV_NAME = venv
PYTHON = python3
MAIN_SCRIPT = main.py

install:
	$(PYTHON) -m venv $(VENV_NAME)
	. $(VENV_NAME)/bin/activate && \
	pip install --upgrade pip && \
	pip install -r constraints.txt && \
	pip install -r requirements.txt

run:
	. $(VENV_NAME)/bin/activate && $(PYTHON) $(MAIN_SCRIPT)

clean:
	rm -rf $(VENV_NAME)

.PHONY: create_environment requirements
PYTHON="python3.8"

create_environment:
	$(PYTHON) -m pip install -q virtualenv
	virtualenv -p `which $(PYTHON)` env

requirements:
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PYTHON) -m pip install -r requirements.txt

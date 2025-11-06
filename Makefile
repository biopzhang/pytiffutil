PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
SCRIPT := tiffutils.py
INSTALL_DIR ?= $(BIN)
LINK_NAME ?= pytiffutil
EXTRA ?=

.PHONY: install deps uninstall clean help

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)

deps: $(BIN)/python requirements.txt
	$(BIN)/python -m pip install -r requirements.txt
	@if [ -n "$(EXTRA)" ]; then $(BIN)/python -m pip install $(EXTRA); fi

install: deps
	chmod +x $(SCRIPT)
	ln -sf "$(PWD)/$(SCRIPT)" "$(INSTALL_DIR)/$(LINK_NAME)"
	@echo "Installed $(LINK_NAME) "

uninstall:
	@if [ -L "$(INSTALL_DIR)/$(LINK_NAME)" ]; then rm "$(INSTALL_DIR)/$(LINK_NAME)"; fi
	@echo "Removed $(INSTALL_DIR)/$(LINK_NAME) (if it existed)"

clean:
	rm -rf "$(VENV)"

help:
	@echo "Targets:"
	@echo "  make deps        # create venv and install dependencies"
	@echo "  make install     # install deps and symlink CLI into $(INSTALL_DIR)"
	@echo "  make uninstall   # remove CLI symlink"
	@echo "  make clean       # remove the virtual environment"

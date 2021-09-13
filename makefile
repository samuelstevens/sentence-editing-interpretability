all: fmt lint types test

repo: FORCE
	./make_clean_repo.sh ../clean-repo

fmt: FORCE
	isort paper/ tests/
	black paper/ tests/

types: FORCE
	mypy --strict paper/ tests/

lint: FORCE
	flake8 paper/ tests/

imports: FORCE
	pyimport paper

test: FORCE
	rm -f *.log
	time python -m pytest tests/ -p no:warnings

coverage: FORCE
	python -m pytest tests/ -p no:warnings --cov=paper

FORCE:


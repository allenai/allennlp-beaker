.PHONY : clean
clean :
	rm -rf .pytest_cache/
	rm -rf allennlp_beaker.egg-info/
	rm -rf dist/
	rm -rf build/
	find . | grep -E '(\.mypy_cache|__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf

.PHONY : lint
lint :
	flake8

.PHONY : format
format :
	black --check .

.PHONY : typecheck
typecheck :
	mypy . --ignore-missing-imports --no-strict-optional --no-site-packages

.PHONY : test
test :
	pytest -v --color=yes

PY ?= python3
ZIP ?= dist/spinn_poisson_submission.zip

.PHONY: all test robustness stress convergence figures note zip clean

all: test figures note zip

## Stage 1 + public practice cases, five runs (official evaluator)
test:
	$(PY) benchmark/evaluate.py --submission submission.py --repeats 5 \
		--json results/public_5run.json

## fallback solver, descending grids, N != 129
robustness:
	$(PY) experiments/check_robustness.py

## 20 x 12 hidden-like generalisation sweep (~9 min)
stress:
	$(PY) experiments/run_stress.py --seeds 20 --first-seed 20260101 \
		--out results/stress.json

## error vs resolution, residual, cond(V)
convergence:
	$(PY) experiments/convergence_study.py

figures:
	$(PY) experiments/make_figures.py

## rebuild the 3-page note (needs tectonic or pdflatex)
note:
	cd report && tectonic results_note.tex
	cp report/results_note.pdf .

## bundle the deliverables
zip: note
	bash build_zip.sh

clean:
	rm -rf dist report/*.aux report/*.log report/*.out

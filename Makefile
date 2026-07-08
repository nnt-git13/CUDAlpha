.PHONY: setup test bench bench-forecaster bench-backtester bench-optimizer bench-kernel-sweep \
        aggregate dashboard profile-forecaster profile-backtester \
        docker-build docker-bench docker-dashboard slurm-up fmt clean

setup:
	pip install -r requirements.txt

# CPU-only correctness suite — no GPU required (harness, validation, schema, math).
test:
	python -m pytest -q

bench:
	python -m bench.run_all --workload all

bench-forecaster:
	python -m bench.run_all --workload forecaster

bench-backtester:
	python -m bench.run_all --workload backtester

# H2 study: sweep the MA window to find where the prefix-sum kernel overtakes naive.
bench-kernel-sweep:
	python -m bench.kernel_sweep

bench-optimizer:
	python -m bench.run_all --workload optimizer

aggregate:
	python -m bench.aggregate

# Profiling recipes (require a GPU + Nsight / torch.profiler). See profiling/.
profile-forecaster:
	python -m profiling.profile_forecaster

profile-backtester:
	python -m profiling.profile_backtester

dashboard:
	python dashboard/app.py

docker-build:
	docker build -f docker/Dockerfile -t cudalpha:latest .

docker-bench:
	docker compose -f docker/docker-compose.yml --profile benchmark up

docker-dashboard:
	docker compose -f docker/docker-compose.yml --profile dashboard up

slurm-up:
	@echo "See slurm/README.md — uses an external slurm-docker-cluster compose project."

fmt:
	ruff format . || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f results/*.json results/*.parquet

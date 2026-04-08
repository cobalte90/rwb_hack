build-artifacts:
	python scripts/build_artifacts.py

bootstrap-artifacts:
	python scripts/bootstrap_runtime.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

submission:
	python scripts/make_submission.py --profile latest_lb

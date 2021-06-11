#!/bin/bash

OMP_NUM_THREADS=1 MPLBACKEND=Agg xvfb-run -s "-screen 0 1400x900x24" python -m pytest \
  -v -n8 \
  -k "not rllib" \
  --random-order-seed 1234 \
  --ignore=tutorials/test/notebooks \
  --cov=maze/test \
  --junitxml=maze/test_report.xml \
  --timeout=150 && \
coverage xml && \
coverage html -d coverage_report && \
coverage report && \
cp -R coverage.xml coverage_report maze

exit $?

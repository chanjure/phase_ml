jupyter nbconvert --to notebook --execute src/notebooks/empirical_phase.ipynb
mv scripts/SRBM-scaling_analysis.nbconvert.ipynb src/notebooks/empirical_phase.ipynb
nbstripout src/notebook/empirical_phase.ipynb        


python -m pytest -v --cov-config=.coveragerc --cov=./ test/

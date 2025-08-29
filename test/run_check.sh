jupyter nbconvert --to notebook --execute src/notebooks/empirical_phase.ipynb
mv src/notebooks/empirical_phase.nbconvert.ipynb src/notebooks/empirical_phase.ipynb
nbstripout src/notebooks/empirical_phase.ipynb        

python -m pytest -v --cov-config=.coveragerc --cov=./ test/

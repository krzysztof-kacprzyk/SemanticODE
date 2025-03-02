# No Equations Needed: Learning System Dynamics Without Relying on Closed-Form ODEs

This is the official repository for the paper "No Equations Needed: Learning System Dynamics Without Relying on Closed-Form ODEs"

## Dependencies
You can install all required dependencies using conda and the following command
```
conda env create -n semantic-odes --file environment.yml
```
This will also install `semantic-odes` (the main module) in editable mode.

## Running all experiments
To run all experiments navigate to `experiments` using
```
cd experiments
``` 
and run
```
./run_scripts/Table_3.sh
./run_scripts/Table_7.sh
./run_scripts/Table_8.sh
python run_scripts/sindy_pharma_example.py
python run_scripts/sindy_pharma_example_generalization.py
```

The results will be saved in
```
experiments/results/
```

## Figures and tables
Jupyter notebooks used to create all figures and tables in the paper can be found in `experiments/analysis`.

## Citation

Kacprzyk, K., & van der Schaar, M. (2025). No Equations Needed: Learning System Dynamics Without Relying on Closed-Form ODEs. The Thirteenth International Conference on Learning Representations.


```
@inproceedings{Kacprzyk.NoEquationsNeeded.2025,
  title = {No {{Equations Needed}}: {{Learning System Dynamics Without Relying}} on {{Closed-Form ODEs}}},
  shorttitle = {No {{Equations Needed}}},
  booktitle = {The {{Thirteenth International Conference}} on {{Learning Representations}}},
  author = {Kacprzyk, Krzysztof and {van der Schaar}, Mihaela},
  year = {2025},
}
```



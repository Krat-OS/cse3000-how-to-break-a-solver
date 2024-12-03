# Cristian Tool README

Create your own notebook and use this to get results:

```python
from result_proccessor import process_results

results = process_results("/path/to/SharpVelvet/out")

for experiment_name, (df, cnf_files) in results.items():
    print(f"\nExperiment: {experiment_name}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Number of CNF files: {len(cnf_files)}")
    
```

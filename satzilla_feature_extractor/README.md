# Satzilla Feature Extractor 2024

**Submodule needs to be setup and paths to features binary need to be changed accordingly.**

This repository contains two scripts designed for feature extraction from SAT instances:

- **`compute_sat_feature_data.py`**: Intended for use in Jupyter Notebooks.
- **`compute_sat_feature_data_cli.py`**: A command-line interface (CLI) version for terminal execution.

---

## How to Use the CLI

Run the CLI script using:

`python compute_sat_feature_data_cli.py <command> [arguments]`

---

### **Available Commands and Arguments**

| Command                       | Arguments to Provide                            |
|------------------------------|-------------------------------------------------|
| `run_python_script`           | `script_path --kwargs key=value ...`            |
| `clear_and_create_directory`  | `dir_path`                                      |
| `compute_features`            | `cnf_dir features_output_dir satzilla_path`    |
| `process_csv_files`           | `features_output_dir`                           |
| `delete_old_instances`        | `directory`                                     |

---

### **Examples**

1. **Run a Python Script:**
   ```python
   python satzilla_feature_extractor/compute_sat_feature_data_cli.py run_python_script /path/to/script.py --kwargs key1=value1 key2=value2
   ```

   This method is intended to be used for:
   
   - generating instances: 
   ```python
   python satzilla_feature_extractor/compute_sat_feature_data_cli.py run_python_script     /path/to/cse3000-how-to-break-a-solver/SharpVelvet/src/generate_instances.py     --kwargs generators=/path/to/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/example_generator_config_mc.json
   ```
   - fuzzing: 
   ```python
   python satzilla_feature_extractor/compute_sat_feature_data_cli.py run_python_script     /path/to/cse3000-how-to-break-a-solver/SharpVelvet/src/run_fuzzer.py     --kwargs counters=/path/to/cse3000-how-to-break-a-solver/SharpVelvet/tool-config/example_counter_config_mc.json     instances=/path/to/cse3000-how-to-break-a-solver/SharpVelvet/out/instances/cnf
   ```

2. **Compute Features:**<br>
    This method produces a CSVs under `SharpVelvet/out/features_output`. First argument is the base of your cnf-s generated with our generators. Second argument set to `SharpVelvet/out/features_output`, will be the location of generated CSVs. For every cnf formula features are computed and stored in a CSV.
   ```python
   python satzilla_feature_extractor/compute_sat_feature_data_cli.py compute_features     /path/to/cse3000-how-to-break-a-solver/SharpVelvet/out/instances/cnf     /path/to/cse3000-how-to-break-a-solver/SharpVelvet/out/features_output     /path/to/revisiting_satzilla/SAT-features-competition2024/features
   ```

3. **Process CSV Files:**<br>
    This method scans a specified directory for CSV files and removes duplicate rows based on the first row's values. It then saves the cleaned CSV files back to the original directory, ensuring no header is included in the output.
   ```python
   python satzilla_feature_extractor/compute_sat_feature_data_cli.py process_csv_files     /path/to/cse3000-how-to-break-a-solver/SharpVelvet/out/features_output
   ```

4. **Delete Old Instances:**
    This method is used when you want to delete instances you already have and generate new ones.
   ```python
   python satzilla_feature_extractor/compute_sat_feature_data_cli.py delete_old_instances     /path/to/cse3000-how-to-break-a-solver/SharpVelvet/out/instances
   ```

---

### **How to Use in Jupyter Notebook**

1. **Import the Script:**
   `import compute_sat_feature_data as csfd`

2. **Use the Functions:**
   - **Generate Instances:**  
     `csfd.run_python_script("/path/to/generate_instances.py", generators="/path/to/generator_config.json")`

   - **Run Fuzzer:**  
     `csfd.run_python_script("/path/to/run_fuzzer.py", counters="/path/to/counter_config.json", instances="/path/to/instances/cnf")`

   - **Compute Features:**  
     `csfd.compute_features("/path/to/instances/cnf", "/path/to/features_output", "/path/to/satzilla/features")`

   - **Process CSV Files:**  
     `csfd.process_csv_files("/path/to/features_output")`

   - **Delete Old Instances:**  
     `csfd.delete_old_instances("/path/to/instances")`

These methods work the same as in the CLI but are directly callable in a Jupyter Notebook environment.
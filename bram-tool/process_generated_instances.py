import os
import subprocess
import glob
import pandas as pd

instances_list = "../SharpVelvet/out/2024-11-27_s764342644147327724_generated_instances.txt"
output_dir = "out/features_output/"
satzilla_path = "SAT-features-competition2024/features"

csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
for csv_file in csv_files:
    os.remove(csv_file)

generated_csv_files = []

with open(instances_list, 'r') as cnf_file:
    for i, line in enumerate(cnf_file):
        CNF_FILE = line.strip()

        generator_type = os.path.basename(CNF_FILE).split('_')[0]
        OUTPUT_FILE = os.path.join(output_dir, f"features_output_{generator_type}.csv")

        subprocess.run([satzilla_path, '-base', CNF_FILE, OUTPUT_FILE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Computed features for instance #" + str(i + 1))

        if OUTPUT_FILE not in generated_csv_files:
            generated_csv_files.append(OUTPUT_FILE)


for csv_file in generated_csv_files:
    df = pd.read_csv(csv_file, header=None)
    first_row = df.iloc[0]
    df = df[df.ne(first_row).any(axis=1)]
    df.iloc[0] = first_row
    df.to_csv(csv_file, index=False, header=False)

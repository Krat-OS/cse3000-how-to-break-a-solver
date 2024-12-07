import os
import shutil
import subprocess
import pandas as pd
import argparse


def run_python_script(script_path: str, **kwargs):
    cmd = ["python3", script_path]
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    print(f"Running script: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def clear_and_create_directory(dir_path: str):
    if os.path.exists(dir_path):
        print(f"Deleting existing directory: {dir_path}")
        shutil.rmtree(dir_path)
    print(f"Creating directory: {dir_path}")
    os.makedirs(dir_path, exist_ok=True)


def compute_features(cnf_dir: str, features_output_dir: str, satzilla_path: str):
    clear_and_create_directory(features_output_dir)
    for cnf_file in os.listdir(cnf_dir):
        if cnf_file.endswith(".cnf"):
            generator_type = cnf_file.split("_")[0]
            output_file = os.path.join(features_output_dir, f"features_output_{generator_type}.csv")
            print(f"Computing features for: {cnf_file}")
            subprocess.run([satzilla_path, "-base", os.path.join(cnf_dir, cnf_file), output_file], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Features saved to: {output_file}")


def process_csv_files(features_output_dir: str):
    for csv_file in os.listdir(features_output_dir):
        if csv_file.endswith(".csv"):
            csv_file_path = os.path.join(features_output_dir, csv_file)
            try:
                print(f"Processing CSV file: {csv_file_path}")
                df = pd.read_csv(csv_file_path, header=None)
                first_row = df.iloc[0]
                df = df[df.ne(first_row).any(axis=1)]
                df.iloc[0] = first_row
                df.to_csv(csv_file_path, index=False, header=False)
                print(f"Processed successfully: {csv_file_path}")
            except Exception as e:
                print(f"Error processing {csv_file_path}: {e}")


def delete_old_instances(directory: str):
    if os.path.exists(directory):
        print(f"Deleting existing instances directory: {directory}")
        shutil.rmtree(directory)
    else:
        print(f"No existing instances directory found at: {directory}")


class CustomHelpFormatter(argparse.HelpFormatter):
    def format_help(self):
        help_text = (
            "Available Commands            Arguments to Provide\n"
            "---------------------        ----------------------\n"
            "run_python_script            script_path --kwargs key=value ...\n"
            "clear_and_create_directory   dir_path\n"
            "compute_features             cnf_dir features_output_dir satzilla_path\n"
            "process_csv_files            features_output_dir\n"
            "delete_old_instances         directory\n"
        )
        return help_text


def main():
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=CustomHelpFormatter,
        add_help=False,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_run = subparsers.add_parser("run_python_script", add_help=False)
    parser_run.add_argument("script_path", help="Path to the Python script")
    parser_run.add_argument("--kwargs", nargs="*", help="Optional key=value arguments")

    parser_clear = subparsers.add_parser("clear_and_create_directory", add_help=False)
    parser_clear.add_argument("dir_path", help="Path to the directory")

    parser_compute = subparsers.add_parser("compute_features", add_help=False)
    parser_compute.add_argument("cnf_dir", help="Path to the CNF files directory")
    parser_compute.add_argument("features_output_dir", help="Path to the features output directory")
    parser_compute.add_argument("satzilla_path", help="Path to the Satzilla executable")

    parser_process = subparsers.add_parser("process_csv_files", add_help=False)
    parser_process.add_argument("features_output_dir", help="Path to the features output directory")

    parser_delete = subparsers.add_parser("delete_old_instances", add_help=False)
    parser_delete.add_argument("directory", help="Path to the instances directory")

    args = parser.parse_args()

    if args.command == "run_python_script":
        run_python_script(args.script_path, **dict(arg.split("=") for arg in args.kwargs or []))
    elif args.command == "clear_and_create_directory":
        clear_and_create_directory(args.dir_path)
    elif args.command == "compute_features":
        compute_features(args.cnf_dir, args.features_output_dir, args.satzilla_path)
    elif args.command == "process_csv_files":
        process_csv_files(args.features_output_dir)
    elif args.command == "delete_old_instances":
        delete_old_instances(args.directory)


if __name__ == "__main__":
    main()

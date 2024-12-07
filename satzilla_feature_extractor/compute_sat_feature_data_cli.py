import argparse
from compute_sat_feature_data import (
    run_python_script,
    clear_and_create_directory,
    compute_features,
    process_csv_files,
    delete_old_instances
)

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

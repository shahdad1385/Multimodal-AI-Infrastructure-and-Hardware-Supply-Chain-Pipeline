import subprocess
import sys
import os

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_samples", "caf_database.db")


def get_python():
    venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def main():
    python = get_python()
    print(f"Using Python: {python}")

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing DB: {DB_PATH}")

    scripts = [
        ("load_data.py", "Loading raw data into DB"),
        ("feature_engineering.py", "Engineering features"),
        ("preprocess.py", "Preprocessing (split, nulls, normalize)"),
    ]

    for script_name, desc in scripts:
        print(f"\n{'=' * 50}")
        print(f"Running: {desc}")
        print(f"{'=' * 50}")
        script_path = os.path.join(SCRIPTS_DIR, script_name)
        subprocess.run([python, script_path], check=True)

    print(f"\n{'=' * 50}")
    print("Pipeline complete!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()

"""
Environment Verification Script for MachineMind AI (ml-service)

Checks for required packages and prints their installed versions.
"""

import sys

REQUIRED_PACKAGES = [
    ("NumPy", "numpy"),
    ("pandas", "pandas"),
    ("Matplotlib", "matplotlib"),
    ("Seaborn", "seaborn"),
    ("scikit-learn", "sklearn"),
    ("Jupyter (notebook / ipykernel)", "IPython"),
]


def check_environment():
    print("=" * 60)
    print("MachineMind AI - ML Environment Check")
    print(f"Python Version: {sys.version.split()[0]} ({sys.executable})")
    print("=" * 60)

    all_passed = True

    for display_name, module_name in REQUIRED_PACKAGES:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "Installed (no __version__ attribute)")
            print(f"  [OK]      {display_name:<30} Version: {version}")
        except ImportError:
            print(f"  [MISSING] {display_name:<30} NOT INSTALLED")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print(" SUCCESS: All required libraries are installed and ready!")
    else:
        print(" WARNING: Some required libraries are missing.")
        print(" Run: pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    check_environment()

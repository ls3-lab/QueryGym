import subprocess
import sys


def test_import_without_optional_dependencies():
    # A plain `pip install querygym` has no pandas/numpy (only the repro and all extras do)
    code = (
        "import sys\n"
        "for m in ('pandas', 'numpy'):\n"
        "    sys.modules[m] = None\n"
        "import querygym, querygym.cli\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

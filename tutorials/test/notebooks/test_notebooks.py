"""
Tests getting started notebooks.
"""

import subprocess
import tempfile
import os
import pytest


_notebook_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../notebooks/getting_started/')


@pytest.mark.parametrize(
    "path",
    [
        os.path.join(_notebook_dir_path, fname) for fname in os.listdir(_notebook_dir_path)
        if os.path.isfile(os.path.join(_notebook_dir_path, fname)) and fname.endswith(".ipynb")
    ]
)
def test_getting_started_notebook(path: str):
    """
    Executes "Getting Started" notebooks.
    :param path: Path to notebook to run.
    """

    # Jupyter's nbconvert converts to e.g. Python and executes before that. The execution returns 0 if the notebook runs
    # successfully and 1 otherwise. We redirect the converted notebook into a temporary directory, since we have no
    # futher interest in it (executing without converting is apparently not possible as of 2021-05-28).
    if "_2" in path:
        with tempfile.TemporaryDirectory() as dirpath:
            output = subprocess.run(
                [
                    'jupyter', 'nbconvert', "--execute", path, "--to", "python", "--ExecutePreprocessor.timeout=-1",
                    "--output-dir={od}".format(od=dirpath)
                ],
                capture_output=True
            )

            assert output.returncode == 0, output.stderr.decode("utf-8")

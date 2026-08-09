import json
from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).parents[1]


class TestExamples(unittest.TestCase):
    def test_modeling_script(self):
        namespace = runpy.run_path(
            ROOT / "examples" / "randalo_modeling.py"
        )
        self.assertIsInstance(namespace["main"](), float)

    def test_cvxpy_notebook(self):
        notebook_path = ROOT / "examples" / "cvxpy-example.ipynb"
        notebook = json.loads(notebook_path.read_text())
        namespace = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            exec(compile(source, str(notebook_path), "exec"), namespace)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path

from highss.cli import resolve_run_paths


class RelativePathResolutionTest(unittest.TestCase):
    def test_paths_are_bound_to_launch_directory_before_chdir(self) -> None:
        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                launch = Path(temp_dir)
                (launch / "target.pdb").write_text("ATOM\n")
                (launch / "model.ckpt").write_bytes(b"checkpoint")
                (launch / "ccd.pkl").write_bytes(b"ccd")

                os.chdir(launch)
                args = argparse.Namespace(
                    target_pdb="target.pdb",
                    checkpoint="model.ckpt",
                    ccd_path="ccd.pkl",
                    work_dir="runs/example",
                )

                target, checkpoint, ccd, work_dir = resolve_run_paths(args)

                self.assertEqual(target, (launch / "target.pdb").resolve())
                self.assertEqual(checkpoint, (launch / "model.ckpt").resolve())
                self.assertEqual(ccd, (launch / "ccd.pkl").resolve())
                self.assertEqual(work_dir, (launch / "runs/example").resolve())

                work_dir.mkdir(parents=True)
                os.chdir(work_dir)

                self.assertTrue(target.is_file())
                self.assertTrue(checkpoint.is_file())
                self.assertTrue(ccd.is_file())
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()

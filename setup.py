from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self) -> None:
        root = Path(__file__).parent.resolve()
        subprocess.run(["cargo", "build", "--release", "--lib"], cwd=root, check=True)
        target = root / "target" / "release"
        names = ["librsnd2.so", "librsnd2.dylib", "rsnd2.dll"]
        package_dir = root / "src-python" / "rsnd2"
        package_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = target / name
            if src.exists():
                shutil.copy2(src, package_dir / name)
                break
        super().run()


setup(cmdclass={"build_py": BuildPy})

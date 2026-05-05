"""Build the optional C accelerator for local Orbit Wars simulations."""

from __future__ import annotations

import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    source = here / "orbit_wars_c.c"
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    include = sysconfig.get_paths()["include"]
    output = here / f"orbit_wars_c{suffix}"
    cmd = [
        "gcc",
        "-O3",
        "-fPIC",
        "-shared",
        f"-I{include}",
        str(source),
        "-lm",
        "-o",
        str(output),
    ]
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.check_call(cmd)
    print(output)


if __name__ == "__main__":
    main()


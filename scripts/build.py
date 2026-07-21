from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
BIN_DIR = ROOT / "bin"


def find_cmake() -> Path:
    executable = shutil.which("cmake")
    if executable:
        return Path(executable)

    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if program_files_x86:
        visual_studio = Path(program_files_x86) / "Microsoft Visual Studio" / "2022"
        for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
            candidate = (
                visual_studio
                / edition
                / "Common7"
                / "IDE"
                / "CommonExtensions"
                / "Microsoft"
                / "CMake"
                / "CMake"
                / "bin"
                / "cmake.exe"
            )
            if candidate.is_file():
                return candidate

    raise RuntimeError("CMake was not found in PATH or Visual Studio 2022.")


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    if platform.system() != "Windows":
        raise RuntimeError("The prebuilt physics backend currently supports Windows only.")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Building the supported extension requires Python 3.10.")
    if sys.maxsize <= 2**32:
        raise RuntimeError("A 64-bit Python installation is required.")
    cmake = str(find_cmake())

    run(
        [
            cmake,
            "-S",
            str(ROOT),
            "-B",
            str(BUILD_DIR),
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
    )
    run([cmake, "--build", str(BUILD_DIR), "--config", "Release"])
    run([cmake, "--install", str(BUILD_DIR), "--config", "Release", "--prefix", str(ROOT)])

    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python did not report an extension module suffix.")

    extension = BIN_DIR / f"_running_env{extension_suffix}"
    if not extension.is_file():
        raise RuntimeError(f"Build completed without producing {extension}.")

    print(f"\nBuilt {extension.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

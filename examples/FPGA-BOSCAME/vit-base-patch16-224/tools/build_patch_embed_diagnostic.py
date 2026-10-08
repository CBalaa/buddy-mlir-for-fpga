#!/usr/bin/env python3
"""Build an isolated patch-embedding image with per-grid UART markers."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "triton"))
from cases import describe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new, empty diagnostic build directory")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["VIT_TRITON_BUILD_ROOT"] = str(output)

    import build as vitbuild
    from export import compile_case

    vitbuild.BUILD_ROOT = output
    case = describe("patch_embed_196x768")
    original_directory = Path(case["directory"])
    config = vitbuild.get_config(case, [])
    runtime_objects = []
    variables = config["variables"]
    objects = vitbuild.shlex.split(variables["RUNTIME_OBJS"])
    vitbuild.command(["make", "--no-print-directory", "-s", "-C", original_directory, *objects])
    runtime_objects = [original_directory / item for item in objects]

    compile_case(case, output)
    adapter_path = output / case["name"] / "adapter.c"
    adapter = adapter_path.read_text()
    call = re.compile(r"(?m)^(\s*)" + re.escape(case["symbol"]) + r"\(([^\n]*)\);$")
    matches = list(call.finditer(adapter))
    if len(matches) != 1:
        raise RuntimeError(f"expected one generated grid call, found {len(matches)}")
    indent, arguments = matches[0].groups()
    traced = (f'{indent}{{\n'
              f'{indent}nr_puts("[patch-grid] BEGIN x=");\n'
              f'{indent}nr_hex32((uint32_t)x); nr_puts("\\r\\n");\n'
              f'{indent}{case["symbol"]}({arguments});\n'
              f'{indent}nr_puts("[patch-grid] END x=");\n'
              f'{indent}nr_hex32((uint32_t)x); nr_puts("\\r\\n");\n'
              f'{indent}}}')
    adapter_path.write_text(adapter[:matches[0].start()] + traced + adapter[matches[0].end():])

    source_directory = output / "diagnostic-source" / case["name"]
    source_directory.mkdir(parents=True)
    source = original_directory / "launch.c"
    launch = source.read_text()
    invocation = "  _mlir_ciface_kernel_patch_embed_196x768(&X,&Weight,&Bias,&Out);"
    if launch.count(invocation) != 1:
        raise RuntimeError("cannot find the patch embedding launch invocation")
    launch = launch.replace(invocation,
        '  nr_puts("[patch-launch] BEGIN kernel\\r\\n");\n' + invocation +
        '\n  nr_puts("[patch-launch] END kernel\\r\\n");')
    (source_directory / "launch.c").write_text(launch)
    shutil.copy2(original_directory / "metadata.json", source_directory / "metadata.json")
    diagnostic_case = dict(case, directory=str(source_directory),
                           launch_source=str(source_directory / "launch.c"))
    run_command = vitbuild.command
    def command_with_original_toolchain_cwd(words, cwd=None, output=None, input_text=None):
        if cwd is not None and Path(cwd) == source_directory:
            cwd = original_directory
        return run_command(words, cwd=cwd, output=output, input_text=input_text)
    vitbuild.command = command_with_original_toolchain_cwd
    result = vitbuild.build_case(diagnostic_case, config, False, runtime_objects)
    record = {"status": "BUILT_STATIC", "diagnostic": "patch_embed_196x768 per-grid begin/end UART markers",
              "build": result, "adapter_sha256": vitbuild.sha(adapter_path),
              "launch_source": str(source_directory / "launch.c")}
    (output / "diagnostic.json").write_text(json.dumps(record, indent=2) + "\n")
    print(output / case["name"] / "nr" / (case["name"] + ".bin"), flush=True)


if __name__ == "__main__":
    main()

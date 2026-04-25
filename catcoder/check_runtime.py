import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from dotenv import dotenv_values, load_dotenv


ROOT = Path(__file__).resolve().parent


def print_result(name: str, ok: bool, detail: str = '') -> bool:
    status = 'PASS' if ok else 'FAIL'
    suffix = f' - {detail}' if detail else ''
    print(f'[{status}] {name}{suffix}')
    return ok


def command_result(cmd: list[str], cwd=ROOT, timeout=10) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except FileNotFoundError:
        return False, 'not on PATH'
    except subprocess.TimeoutExpired:
        return False, f'timed out after {timeout}s'

    output = (proc.stdout or proc.stderr or '').strip().splitlines()
    detail = output[0] if output else f'exit code {proc.returncode}'
    return proc.returncode == 0, detail


def python_snippet(snippet: str, cwd=ROOT, timeout=120) -> tuple[bool, str]:
    return command_result([sys.executable, '-c', snippet], cwd=cwd, timeout=timeout)


def check_python_deps() -> bool:
    ok, detail = python_snippet(
        "import pyarrow; "
        "pyarrow.PyExtensionType = getattr(pyarrow, 'PyExtensionType', pyarrow.ExtensionType); "
        "import datasets, openai, dotenv, backoff; "
        "print('pyarrow=' + pyarrow.__version__)"
    )
    return print_result('Python dependencies', ok, detail)


def check_datasets() -> bool:
    ok, detail = python_snippet(
        "import pyarrow; "
        "pyarrow.PyExtensionType = getattr(pyarrow, 'PyExtensionType', pyarrow.ExtensionType); "
        "from datasets import load_from_disk; "
        "print('java=' + str(len(load_from_disk('java/dataset/javaeval'))) + "
        "' rust=' + str(len(load_from_disk('rust/dataset/rusteval'))))"
    )
    return print_result('Benchmark datasets', ok, detail)


def check_env() -> bool:
    env_file = ROOT / '.env'
    values = dotenv_values(env_file) if env_file.exists() else {}
    keys = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL']
    missing = [key for key in keys if not values.get(key) and not os.environ.get(key)]
    if missing:
        return print_result('Cloud model env', False, 'missing ' + ', '.join(missing))
    model = values.get('OPENAI_MODEL') or os.environ.get('OPENAI_MODEL')
    base_url = values.get('OPENAI_BASE_URL') or os.environ.get('OPENAI_BASE_URL')
    return print_result('Cloud model env', True, f'model={model} base_url={base_url}')


def check_java_toolchain() -> bool:
    java_home = os.environ.get('JAVA_HOME')
    java_bin = Path(java_home) / 'bin' / 'java' if java_home else None
    if java_bin and java_bin.exists():
        java_ok, java_detail = command_result([str(java_bin), '-version'])
    else:
        java_ok = False
        java_path = shutil.which('java')
        java_detail = 'JAVA_HOME/bin/java not found'
        if java_path:
            java_detail += f' (java shim/path: {java_path})'
    d4j_path = shutil.which('defects4j')
    d4j_ok = d4j_path is not None
    print_result('Java runtime', java_ok, java_detail)
    print_result('Defects4J', d4j_ok, d4j_path or 'not on PATH')
    return java_ok and d4j_ok


def check_rust_toolchain() -> bool:
    cargo_path = shutil.which('cargo')
    analyzer_path = shutil.which('rust-analyzer')
    crates_dir = ROOT / 'rust' / 'crates'
    print_result('Cargo', cargo_path is not None, cargo_path or 'not on PATH')
    print_result('rust-analyzer', analyzer_path is not None, analyzer_path or 'not on PATH')
    print_result('Rust crates', crates_dir.is_dir(), str(crates_dir))
    return cargo_path is not None and analyzer_path is not None and crates_dir.is_dir()


def check_cloud_smoke() -> bool:
    snippet = (
        "from inference import OpenAIModel; "
        "print(repr(OpenAIModel(max_new_tokens=128, temp=0.1).infer('Return exactly: ok').strip()))"
    )
    java_ok, java_detail = python_snippet(snippet, cwd=ROOT / 'java')
    rust_ok, rust_detail = python_snippet(snippet, cwd=ROOT / 'rust')
    print_result('Java cloud smoke', java_ok, java_detail)
    print_result('Rust cloud smoke', rust_ok, rust_detail)
    return java_ok and rust_ok


def check_codegen_smoke() -> bool:
    java_snippet = """
import pyarrow
pyarrow.PyExtensionType = getattr(pyarrow, 'PyExtensionType', pyarrow.ExtensionType)
from datasets import load_from_disk
from inference import OpenAIModel
from util import build_prompt, truncate_generation, remove_markdown, fix_fragmented_code
row = load_from_disk('./dataset/javaeval')[0]
payload = {
    'focal_fn_signature': row['focal_fn_signature'],
    'docstring': row['docstring'],
    'focal_ctx': row['extended_context'],
    'rag_data': row['rag_data'],
}
code = OpenAIModel(max_new_tokens=768, temp=0.1).infer(build_prompt(payload, True)).strip()
for fn in (truncate_generation, remove_markdown, fix_fragmented_code):
    code = fn(code)
if code and not code.startswith(('public', 'private', 'protected', 'static', '@')):
    code = row['focal_fn_signature'] + ' ' + code
print('task_id=' + row['task_id'] + ' chars=' + str(len(code)))
print(code[:500] if code else '<empty>')
"""
    rust_snippet = """
import pyarrow
pyarrow.PyExtensionType = getattr(pyarrow, 'PyExtensionType', pyarrow.ExtensionType)
from datasets import load_from_disk
from inference import OpenAIModel
from util import build_prompt, truncate_generation, remove_markdown, fix_fragmented_code
row = load_from_disk('./dataset/rusteval')[0]
payload = {
    'focal_fn_signature': row['signature'],
    'docstring': row['docstring'],
    'focal_ctx': row['extended_context'],
    'rag_data': row['rag_data'],
}
code = OpenAIModel(max_new_tokens=768, temp=0.1).infer(build_prompt(payload, True)).strip()
for fn in (truncate_generation, remove_markdown, fix_fragmented_code):
    code = fn(code)
if code and not code.startswith(('fn', 'pub fn')):
    code = row['signature'] + ' ' + code
print('task_id=' + str(row['task_id']) + ' chars=' + str(len(code)))
print(code[:500] if code else '<empty>')
"""
    java_ok, java_detail = python_snippet(java_snippet, cwd=ROOT / 'java', timeout=180)
    rust_ok, rust_detail = python_snippet(rust_snippet, cwd=ROOT / 'rust', timeout=180)
    print_result('Java codegen smoke', java_ok, java_detail)
    print_result('Rust codegen smoke', rust_ok, rust_detail)
    return java_ok and rust_ok


def main() -> int:
    parser = argparse.ArgumentParser(description='Check CatCoder runtime readiness.')
    parser.add_argument('--cloud-smoke', action='store_true', help='call the configured cloud model once per language')
    parser.add_argument('--codegen-smoke', action='store_true', help='run one Java and one Rust code generation sample')
    args = parser.parse_args()

    load_dotenv(ROOT / '.env', override=False)

    required_ok = [
        check_python_deps(),
        check_datasets(),
        check_env(),
    ]
    java_ready = check_java_toolchain()
    rust_ready = check_rust_toolchain()

    optional_ok = []
    if args.cloud_smoke:
        optional_ok.append(check_cloud_smoke())
    if args.codegen_smoke:
        optional_ok.append(check_codegen_smoke())

    print('')
    print('Summary:')
    print_result('Python/cloud prerequisites', all(required_ok))
    print_result('Java full evaluation readiness', java_ready)
    print_result('Rust full evaluation readiness', rust_ready)
    if optional_ok:
        print_result('Requested smoke checks', all(optional_ok))

    return 0 if all(required_ok) and all(optional_ok or [True]) else 1


if __name__ == '__main__':
    raise SystemExit(main())

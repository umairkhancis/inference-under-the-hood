import pathlib

def folder_size(path):
    p = pathlib.Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

def format_size(nbytes):
    if nbytes < 1024**2:
        return f"{nbytes/1024:.1f} KB"
    if nbytes < 1024**3:
        return f"{nbytes/1024**2:.1f} MB"
    return f"{nbytes/1024**3:.2f} GB"

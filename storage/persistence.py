import os
from pathlib import Path

def atomic_write(path: Path, data: bytes):
    """
    Writes data to a file atomically by first writing to a temporary file
    and then renaming it. This prevents partial writes in case of a crash.
    """
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    try:
        with open(tmp_path, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        
        os.rename(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def ensure_dir(path: Path):
    """Creates a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)

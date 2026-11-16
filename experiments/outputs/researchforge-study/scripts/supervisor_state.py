"""Atomic supervisor JSON writes with bounded Windows sharing-conflict retries."""
import json
import os
from pathlib import Path
import tempfile
import time


def atomic_state(path,value,attempts=30):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix=path.stem+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        for attempt in range(attempts):
            try:
                os.replace(temporary,path);return
            except PermissionError:
                if attempt+1==attempts:raise
                time.sleep(min(.1*(attempt+1),1.))
    finally:
        if temporary.exists():temporary.unlink()

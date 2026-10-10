import hashlib
import json
import re
import shutil
import threading
from pathlib import Path
from typing import Protocol
import joblib


class ModelRepository(Protocol):
    def load(self, name: str): ...


class ModelIntegrityError(ValueError):
    """A stored model does not match its manifest."""


class FileModelRegistry:
    """Repository adapter: immutable versions and atomic active pointers."""

    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}
        self.lock = threading.RLock()

    @staticmethod
    def safe(value):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
            raise ValueError("Invalid model identifier.")
        return value

    def register(self, name, version, source):
        name, version = self.safe(name), self.safe(version)
        source = Path(source)
        bundle = joblib.load(source / "model.joblib")
        target = self.root / name / version
        with self.lock:
            target.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(source / "model.joblib", target / "model.joblib")
            report = json.loads((source / "metrics.json").read_text())
            manifest = {
                "name": name,
                "version": version,
                "features": bundle["features"],
                "sha256": hashlib.sha256((target / "model.joblib").read_bytes()).hexdigest(),
                "selected_model": report["selected_model"],
                "calibration": report["calibration"],
                "data": report["data"],
                "task": report["task"],
                "event_rate": report["event_rate"],
            }
            (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
            shutil.copyfile(source / "metrics.json", target / "metrics.json")
        return manifest

    def promote(self, name, version):
        name, version = self.safe(name), self.safe(version)
        manifest = json.loads((self.root / name / version / "manifest.json").read_text())
        self._verified_bundle(self.root / name / version, manifest)
        with self.lock:
            temporary = self.root / name / "active.tmp"
            temporary.write_text(json.dumps({"version": version}))
            temporary.replace(self.root / name / "active.json")

    def _verified_bundle(self, path, manifest):
        data = (path / "model.joblib").read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
            raise ModelIntegrityError("Model integrity check failed.")
        bundle = joblib.load(path / "model.joblib")
        if bundle["features"] != manifest["features"]:
            raise ModelIntegrityError("Model schema mismatch.")
        return bundle

    def load(self, name):
        name = self.safe(name)
        with self.lock:
            version = self.safe(json.loads((self.root / name / "active.json").read_text())["version"])
            key = (name, version)
            if key not in self.cache:
                directory = self.root / name / version
                manifest = json.loads((directory / "manifest.json").read_text())
                self.cache[key] = (self._verified_bundle(directory, manifest), manifest)
            return self.cache[key]

    def names(self):
        return sorted(p.parent.name for p in self.root.glob("*/active.json"))

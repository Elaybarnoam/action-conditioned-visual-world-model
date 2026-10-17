"""Dependency-free release demo, checksums, and verified artifact downloads."""

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/release-manifest.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_files(root, entries):
    for name, expected in entries.items():
        path = (Path(root) / name).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("Artifact path escapes its directory")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    return len(entries)


def unpack_verified(archive, destination):
    """Reject traversal and symlinks before extracting any untrusted archive member."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            path = (destination / member.filename).resolve()
            if (
                not path.is_relative_to(destination)
                or (member.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Unsafe artifact archive member")
        z.extractall(destination)


def fetch(bundle, destination=None):
    manifest = read(MANIFEST)
    entry = manifest["assets"][bundle]
    destination = Path(destination or ROOT / "data/release" / bundle)
    destination.mkdir(parents=True, exist_ok=True)
    installed = destination / "bundle-manifest.json"
    expected_contents = {k: v for k, v in entry["files"].items() if k != "bundle-manifest.json"}
    if installed.exists() and read(installed)["files"] == expected_contents:
        verify_files(destination, entry["files"])
        print(f"Verified installed {bundle}: {destination}")
        return destination
    with tempfile.TemporaryDirectory(prefix="world-model-download-") as temporary:
        archive = Path(temporary) / entry["filename"]
        request = urllib.request.Request(
            entry["url"], headers={"User-Agent": "visual-world-model-v1.0.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as file:
            shutil.copyfileobj(response, file)
        if archive.stat().st_size != entry["bytes"] or digest(archive) != entry["sha256"]:
            raise ValueError("Downloaded bundle checksum/size mismatch")
        staging = Path(temporary) / "unpacked"
        unpack_verified(archive, staging)
        verify_files(staging, entry["files"])
        for path in staging.rglob("*"):
            if path.is_file():
                target = destination / path.relative_to(staging)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    print(f"Downloaded and verified {bundle}: {destination}")
    return destination


def make_demo_server(port=8768):
    examples = read(ROOT / "results/demo/examples.json")
    info = read(ROOT / "results/demo/info.json")
    page = (ROOT / "src/result_viewer.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path)
            try:
                if route.path == "/":
                    return self.reply(200, "text/html; charset=utf-8", page)
                if route.path == "/api/info":
                    value = info
                elif route.path == "/api/example":
                    params = parse_qs(route.query, strict_parsing=True)
                    if set(params) != {"episode", "frame"} or any(
                        len(v) != 1 for v in params.values()
                    ):
                        raise ValueError("Supply one example index and frame=0")
                    index = int(params["episode"][0])
                    if int(params["frame"][0]) != 0 or not 0 <= index < len(examples):
                        raise ValueError("Example index outside the released six examples")
                    value = examples[index]
                else:
                    return self.reply(404, "application/json", b'{"error":"Not found"}')
                self.reply(200, "application/json", json.dumps(value, allow_nan=False).encode())
            except (ValueError, KeyError) as error:
                self.reply(400, "application/json", json.dumps({"error": str(error)}).encode())

        def reply(self, status, content_type, body):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Open exact saved predictions; standard library only")
    demo.add_argument("--port", type=int, default=8768)
    download = sub.add_parser("fetch", help="Download a checksum-pinned release bundle")
    download.add_argument(
        "--bundle", choices=("inference", "training", "evidence"), default="inference"
    )
    download.add_argument("--destination", type=Path)
    sub.add_parser("verify", help="Verify committed results, demo and figures")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.bundle, args.destination)
    elif args.command == "verify":
        count = verify_files(ROOT, read(MANIFEST)["public_files"])
        print(f"Verified {count} published files against SHA-256 manifest.")
    else:
        verify_files(ROOT, read(MANIFEST)["public_files"])
        with make_demo_server(args.port) as server:
            print(f"Frozen results: http://127.0.0.1:{server.server_port}/", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()

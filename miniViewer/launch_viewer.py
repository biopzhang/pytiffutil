import argparse
import json
import os
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class DZIHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path == "/list_dzi":
            dzi_files = []
            tiles_dir = os.path.join(os.getcwd(), "tiles")
            if os.path.isdir(tiles_dir):
                for filename in sorted(os.listdir(tiles_dir)):
                    if filename.endswith(".dzi"):
                        name = os.path.splitext(filename)[0]
                        dzi_files.append({"name": name, "path": f"tiles/{filename}"})

            payload = json.dumps(dzi_files).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        super().do_GET()

    def log_message(self, format, *args):
        pass


def open_browser(url):
    time.sleep(0.5)
    webbrowser.open(url)


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the miniViewer tile viewer.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})")
    parser.add_argument("--viewer", default="viewer.html", choices=["viewer.html", "split_viewer.html"])
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    handler = partial(DZIHandler, directory=root_dir)

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as err:
        raise SystemExit(f"Could not start miniViewer on {args.host}:{args.port}: {err}") from err

    url = f"http://{args.host}:{args.port}/{args.viewer}"
    print(f"Server running at {url}", flush=True)
    print("Press Ctrl-C to stop.", flush=True)

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

"""In-process MJPEG + key-relay HTTP server, so a tool's live feed and
keyboard controls can be embedded in the Phase 1 web UI instead of a
native cv2.imshow window.

Each tool that supports this opts in via --serve PORT; without it, tools
keep working exactly as before (native window, cv2.waitKey) for direct
CLI use -- this is additive, not a replacement.
"""

import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

_BOUNDARY = "frame"
_STREAM_INTERVAL_S = 0.03  # caps the push rate to the browser independent of camera fps


class StreamServer:
    def __init__(self, port: int):
        self._port = port
        self._frame_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._keys: "queue.Queue[int]" = queue.Queue()
        self._params: dict[str, float] = {}  # live-adjustable values (e.g. a sensitivity slider), set via /param
        self._params_lock = threading.Lock()
        self._gesture_requests: "queue.Queue[tuple[str, bool]]" = queue.Queue()  # (action, clear) from /record
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def publish(self, frame_bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._frame_lock:
                self._latest_jpeg = buf.tobytes()

    def poll_key(self) -> int | None:
        try:
            return self._keys.get_nowait()
        except queue.Empty:
            return None

    def get_param(self, name: str, default: float) -> float:
        with self._params_lock:
            return self._params.get(name, default)

    def poll_gesture_request(self) -> tuple[str, bool] | None:
        try:
            return self._gesture_requests.get_nowait()
        except queue.Empty:
            return None

    def start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):  # silence default per-request stderr logging
                pass

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
                    self._cors()
                    self.end_headers()
                    try:
                        while True:
                            with server._frame_lock:
                                jpeg = server._latest_jpeg
                            if jpeg is not None:
                                self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                                self.wfile.write(jpeg)
                                self.wfile.write(b"\r\n")
                            time.sleep(_STREAM_INTERVAL_S)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                elif path == "/key":
                    qs = parse_qs(urlparse(self.path).query)
                    code = qs.get("code", [None])[0]
                    if code is not None:
                        server._keys.put(int(code))
                    self.send_response(204)
                    self._cors()
                    self.end_headers()
                elif path == "/param":
                    qs = parse_qs(urlparse(self.path).query)
                    name = qs.get("name", [None])[0]
                    value = qs.get("value", [None])[0]
                    if name is not None and value is not None:
                        with server._params_lock:
                            server._params[name] = float(value)
                    self.send_response(204)
                    self._cors()
                    self.end_headers()
                elif path == "/record":
                    qs = parse_qs(urlparse(self.path).query)
                    action = qs.get("action", [None])[0]
                    clear = qs.get("clear", ["0"])[0] == "1"
                    if action is not None:
                        server._gesture_requests.put((action, clear))
                    self.send_response(204)
                    self._cors()
                    self.end_headers()
                else:
                    self.send_response(404)
                    self._cors()
                    self.end_headers()

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self._port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()

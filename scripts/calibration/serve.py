"""临时验证用的静态文件服务：给 tmp/faces 下的测试图加上 CORS 头。

浏览器里用 fetch 拉图必须过 CORS，python 自带的 http.server 不发这个头，
所以这里包一层。只在本地验证时用，不属于项目代码。
"""

import functools
import http.server
import socketserver
import sys


class CorsHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    directory = sys.argv[2] if len(sys.argv) > 2 else "."
    handler = functools.partial(CorsHandler, directory=directory)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {directory} on http://127.0.0.1:{port}", flush=True)
        httpd.serve_forever()

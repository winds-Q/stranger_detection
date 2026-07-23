import logging
import os
import socket

from web import app as web_app

logger = logging.getLogger(__name__)


def is_port_in_use(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        with socket.create_connection((probe_host, port), timeout=0.5):
            return True
    except OSError:
        return False


def main():
    try:
        from waitress import serve
    except ImportError as exc:
        raise SystemExit("缺少 Waitress，请先执行：pip install -r requirements.txt") from exc

    host = os.environ.get("STRANGER_DETECTION_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("STRANGER_DETECTION_WEB_PORT", "5050"))
    if is_port_in_use(host, port):
        raise SystemExit(f"端口 {port} 已被占用，Web 服务可能已经运行")

    retention_worker = web_app.initialize_runtime()
    logger.info("正式 Web 服务已启动: http://%s:%d", host, port)
    try:
        serve(
            web_app.app,
            host=host,
            port=port,
            threads=8,
            channel_timeout=60,
            clear_untrusted_proxy_headers=True,
        )
    finally:
        web_app.stop_runtime(retention_worker)


if __name__ == "__main__":
    main()

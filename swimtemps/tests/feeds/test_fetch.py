import asyncio
import socket
import threading
import time

import httpx
import pytest

from feeds import fetch as fetch_module
from feeds.fetch import MAX_BYTES, FetchError, fetch_json

ADDRESSES = {
    "lake.example": "93.184.216.34",
    "other.example": "151.101.1.1",
    "localhost": "127.0.0.1",
    "sneaky.example": "10.1.2.3",
}


def resolver(host, port, *args, **kwargs):
    if host not in ADDRESSES and not host.replace(".", "").isdigit() and ":" not in host:
        raise socket.gaierror("no such host")
    ip = ADDRESSES.get(host, host)
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 6, "", (ip, port or 0))]


def client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def json_ok(request):
    return httpx.Response(200, json={"labels": [1], "datasets": [{"data": [13.5]}]})


def test_a_public_json_feed_is_fetched():
    assert fetch_json("https://lake.example/temps", client_for(json_ok), resolver) == {
        "labels": [1], "datasets": [{"data": [13.5]}]}


def test_the_user_agent_says_only_swimtemps():
    seen = []

    def handler(request):
        seen.append(request.headers["user-agent"])
        return json_ok(request)

    fetch_json("https://lake.example/temps", client_for(handler), resolver)
    assert seen == ["swimtemps/0.1"]


@pytest.mark.parametrize("url", [
    "http://localhost/temps", "http://127.0.0.1:8000/", "http://10.0.0.5/temps", "http://sneaky.example/",
    "http://169.254.169.254/latest/meta-data/", "http://[::1]/", "http://192.168.1.1/", "http://0.0.0.0/",
])
def test_internal_addresses_are_refused_before_any_request(url):
    requests = []

    def handler(request):
        requests.append(request)
        return json_ok(request)

    with pytest.raises(FetchError, match="private or internal"):
        fetch_json(url, client_for(handler), resolver)
    assert requests == []


@pytest.mark.parametrize("url", ["ftp://lake.example/temps", "file:///etc/passwd", "lake.example/temps"])
def test_only_http_and_https(url):
    with pytest.raises(FetchError, match="http"):
        fetch_json(url, client_for(json_ok), resolver)


def test_an_unknown_host_is_refused():
    with pytest.raises(FetchError, match="nowhere.example"):
        fetch_json("https://nowhere.example/", client_for(json_ok), resolver)


def test_a_redirect_to_an_internal_address_is_not_followed():
    requests = []

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    with pytest.raises(FetchError, match="private or internal"):
        fetch_json("https://lake.example/temps", client_for(handler), resolver)
    assert requests == ["https://lake.example/temps"]


def test_a_redirect_to_another_public_host_is_followed():
    def handler(request):
        if request.url.host == "lake.example":
            return httpx.Response(301, headers={"location": "https://other.example/v2/temps"})
        return json_ok(request)

    assert fetch_json("https://lake.example/temps", client_for(handler), resolver)["labels"] == [1]


def test_a_relative_redirect_is_followed():
    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "/new"})
        return json_ok(request)

    assert fetch_json("https://lake.example/old", client_for(handler), resolver)["labels"] == [1]


def test_four_redirects_are_refused():
    def handler(request):
        n = int(request.url.path.strip("/") or 0)
        return httpx.Response(302, headers={"location": f"/{n + 1}"})

    with pytest.raises(FetchError, match="redirect"):
        fetch_json("https://lake.example/0", client_for(handler), resolver)


def test_three_redirects_are_followed():
    def handler(request):
        n = int(request.url.path.strip("/") or 0)
        return json_ok(request) if n == 3 else httpx.Response(302, headers={"location": f"/{n + 1}"})

    assert fetch_json("https://lake.example/0", client_for(handler), resolver)["labels"] == [1]


def test_a_body_over_five_megabytes_is_refused():
    big = b"[" + b"1," * (3 * 1024 * 1024) + b"1]"
    with pytest.raises(FetchError, match="5 MB"):
        fetch_json("https://lake.example/", client_for(lambda r: httpx.Response(200, content=big)), resolver)


def test_a_streamed_body_over_five_megabytes_is_refused_without_reading_it_all():
    chunks = []

    async def stream():
        for _ in range(100):
            chunks.append(1)
            yield b" " * (256 * 1024)

    handler = lambda r: httpx.Response(200, content=stream())  # noqa: E731
    with pytest.raises(FetchError, match="5 MB"):
        fetch_json("https://lake.example/", client_for(handler), resolver)
    assert len(chunks) * 256 * 1024 <= MAX_BYTES + 256 * 1024


@pytest.fixture
def short_deadline(monkeypatch):
    monkeypatch.setattr(fetch_module, "TIMEOUT_SECONDS", 0.5)


def test_a_trickling_body_is_cut_off_at_the_deadline(short_deadline):
    async def stream():
        for _ in range(100):
            await asyncio.sleep(0.05)
            yield b" "

    handler = lambda r: httpx.Response(200, content=stream())  # noqa: E731
    started = time.monotonic()
    with pytest.raises(FetchError, match="seconds"):
        fetch_json("https://lake.example/", client_for(handler), resolver)
    assert time.monotonic() - started < 1.5


def test_slow_name_resolution_counts_against_the_deadline(short_deadline):
    def slow(host, port, *args, **kwargs):
        time.sleep(2)
        return resolver(host, port)

    started = time.monotonic()
    with pytest.raises(FetchError, match="seconds"):
        fetch_json("https://lake.example/", client_for(json_ok), slow)
    assert time.monotonic() - started < 1.5


@pytest.fixture
def trickling_server():
    """A local server that sends its response headers one byte at a time, forever."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    stop = threading.Event()

    def serve():
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                continue
            with conn:
                for byte in b"HTTP/1.1 200 OK\r\nX-Slow: " + b"a" * 10_000:
                    if stop.is_set():
                        break
                    try:
                        conn.sendall(bytes([byte]))
                    except OSError:
                        break
                    time.sleep(0.05)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield server.getsockname()[1]
    stop.set()
    server.close()


def test_headers_trickled_a_byte_at_a_time_are_cut_off_at_the_deadline(short_deadline, trickling_server):
    def public(host, port, *args, **kwargs):
        # Passes the address check as if public; the connection still goes to the local server.
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    started = time.monotonic()
    with pytest.raises(FetchError, match="seconds"):
        fetch_json(f"http://127.0.0.1:{trickling_server}/temps", resolver=public)
    assert time.monotonic() - started < 2


def test_a_timeout_is_a_fetch_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(FetchError, match="20 seconds"):
        fetch_json("https://lake.example/", client_for(handler), resolver)


def test_a_connection_failure_is_a_fetch_error():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(FetchError, match="Could not connect"):
        fetch_json("https://lake.example/", client_for(handler), resolver)


def test_a_web_page_is_refused():
    page = lambda r: httpx.Response(200, html="<!doctype html><html><body>Live temps</body></html>")  # noqa: E731
    with pytest.raises(FetchError, match="JSON"):
        fetch_json("https://lake.example/", client_for(page), resolver)


def test_an_error_status_is_refused():
    with pytest.raises(FetchError, match="404"):
        fetch_json("https://lake.example/", client_for(lambda r: httpx.Response(404, text="gone")), resolver)

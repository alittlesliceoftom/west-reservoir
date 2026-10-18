"""
Fetching a feed: one guarded GET.

Feed URLs are typed in by people, so every hop is checked before it is
requested: only http and https, and never a host that resolves to a
private, loopback, link-local, multicast or reserved address. Redirects
are followed by hand (at most three) so each new host is checked too. The
body is capped at 5 MB. One 20-second deadline covers the whole fetch:
name lookups, every hop's headers and the body, so a server that trickles
bytes cannot hold the caller. The user-agent names the tool and nothing else.

`fetch_json` is synchronous: it runs its own event loop, so call it from a
thread with no running loop (the worker, or the web's threadpool).
"""

import asyncio
import ipaddress
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from urllib.parse import urljoin, urlparse

import httpx

USER_AGENT = "swimtemps/0.1"
TIMEOUT_SECONDS = 20
MAX_REDIRECTS = 3
MAX_BYTES = 5 * 1024 * 1024

# Lookups run here, not in the loop's default executor: asyncio.run waits for
# that one at exit, and a lookup that outlives the deadline must not.
_LOOKUPS = ThreadPoolExecutor(max_workers=4, thread_name_prefix="feed-dns")


class FetchError(Exception):
    pass


def _refused(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_multicast


async def check_url(url: str, resolver=socket.getaddrinfo) -> None:
    """Raise FetchError unless `url` is http(s) on a host with only public addresses."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("The feed address must start with http:// or https://.")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    lookup = partial(resolver, parts.hostname, port, type=socket.SOCK_STREAM)
    try:
        found = await asyncio.get_running_loop().run_in_executor(_LOOKUPS, lookup)
    except (socket.gaierror, UnicodeError):
        raise FetchError(f"Could not find the host {parts.hostname}.") from None
    addresses = {info[4][0] for info in found}
    if not addresses or any(_refused(address) for address in addresses):
        raise FetchError(f"{parts.hostname} is a private or internal address, so it cannot be a feed.")


async def _read(response: httpx.Response) -> bytes:
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BYTES:
        raise FetchError("The feed is larger than 5 MB.")
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body += chunk
        if len(body) > MAX_BYTES:
            raise FetchError("The feed is larger than 5 MB.")
    return bytes(body)


async def _get(client: httpx.AsyncClient, url: str, resolver) -> object:
    for _ in range(MAX_REDIRECTS + 1):
        await check_url(url, resolver)
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        async with client.stream("GET", url, headers=headers, follow_redirects=False,
                                 timeout=TIMEOUT_SECONDS) as response:
            if response.is_redirect and "location" in response.headers:
                url = urljoin(url, response.headers["location"])
                continue
            if response.status_code != 200:
                raise FetchError(f"The feed answered {response.status_code} instead of 200.")
            body = await _read(response)
        try:
            return json.loads(body)
        except ValueError:
            looks_like = " (it looks like a web page)" if body.lstrip()[:1] == b"<" else ""
            raise FetchError(f"The feed did not return JSON{looks_like}.") from None
    raise FetchError(f"The feed redirected more than {MAX_REDIRECTS} times.")


async def _fetch(url: str, client: httpx.AsyncClient | None, resolver) -> object:
    slow = FetchError(f"The feed took more than {TIMEOUT_SECONDS} seconds to answer.")
    own = client is None
    client = client or httpx.AsyncClient()
    try:
        async with asyncio.timeout(TIMEOUT_SECONDS):
            return await _get(client, url, resolver)
    except TimeoutError:
        raise slow from None
    except httpx.TimeoutException:
        raise slow from None
    except httpx.TransportError as error:
        raise FetchError(f"Could not connect to the feed: {error}.") from None
    finally:
        if own:
            await client.aclose()


def fetch_json(url: str, client: httpx.AsyncClient | None = None, resolver=socket.getaddrinfo) -> object:
    """GET `url` and return its JSON, or raise FetchError with a plain reason."""
    return asyncio.run(_fetch(url, client, resolver))

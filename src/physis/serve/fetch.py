"""Fetching a presigned image URL without becoming an SSRF proxy.

The service is handed a URL by a caller and fetches it. That is a request the
server makes on someone else's instruction, which is the whole shape of server
side request forgery: the container sits inside a network the caller cannot
reach, so `http://169.254.169.254/` asks it to read cloud metadata on their
behalf and hand the result back.

Authentication narrows who can ask. It does not make the request safe, because
the caller with the token is not necessarily the only party who ever holds it.

Four gates, in order, because each one is cheap and the next is not:

1. scheme must be https, so credentials in the query string are not sent in
   clear and a plain-http redirect target cannot be reached
2. hostname must pass the allowlist, when one is configured
3. every resolved address must be public, which is the gate that catches a
   hostname deliberately pointed at 127.0.0.1 or 169.254.169.254
4. the body is read with a hard byte ceiling rather than into memory unbounded

Redirects are refused outright. A redirect is the host asking to change the
destination after the checks have run, and re-running them per hop is more
surface than the feature is worth: a presigned object URL does not redirect.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.error
import urllib.request
from urllib.parse import urlsplit

# 16-bit 384x384 PNG is around 135 KB. The ceiling is generous enough for an
# unprocessed hospital export and small enough that eight of them cannot
# exhaust a 2-core container.
DEFAULT_MAX_BYTES = 32 * 1024 * 1024
CONNECT_TIMEOUT_S = 10.0


class ImageFetchError(Exception):
    """Base for anything that stops a URL becoming bytes."""


class ImageHostRejected(ImageFetchError):
    """Blocked before any connection: scheme, allowlist, or a private address."""


class ImageTooLarge(ImageFetchError):
    """Body exceeded the byte ceiling."""


class ImageUnreachable(ImageFetchError):
    """Connected, or tried to, and did not come back with the object."""


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Turn a redirect into a refusal rather than following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ImageHostRejected(f"redirect to {urlsplit(newurl).hostname} refused")


def allowed_hosts() -> tuple[str, ...]:
    """Hostnames permitted as image sources, from PHYSIS_IMAGE_HOSTS.

    Empty means any public host, which is the weakest of the four gates left
    doing the work. Set it to the R2 hostname in production; leaving it unset
    still blocks private and link-local addresses, so the metadata endpoint is
    unreachable either way.
    """
    raw = os.environ.get("PHYSIS_IMAGE_HOSTS", "")
    return tuple(h.strip().lower() for h in raw.split(",") if h.strip())


def _host_allowed(hostname: str, allowlist: tuple[str, ...]) -> bool:
    """Exact match, or a subdomain of an allowlisted suffix.

    `.r2.dev` permits `bucket.r2.dev` and not `evil-r2.dev`, which a bare
    `endswith` would wave through.
    """
    if not allowlist:
        return True
    host = hostname.lower()
    return any(host == entry or host.endswith("." + entry.lstrip(".")) for entry in allowlist)


def _addresses_are_public(hostname: str) -> None:
    """Resolve, and reject unless every address is publicly routable.

    Every address, not the first: a hostname can resolve to a public address and
    a loopback one, and connecting picks whichever the stack prefers. Checking
    one and connecting to another is a check that proves nothing.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as err:
        raise ImageUnreachable(f"cannot resolve {hostname}") from err

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local  # 169.254.0.0/16 holds the metadata endpoint
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise ImageHostRejected(f"{hostname} resolves to a non-public address")


def _is_loopback(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    return all(ipaddress.ip_address(i[4][0]).is_loopback for i in infos)


def loopback_allowed() -> bool:
    """Whether plain http to 127.0.0.1 is permitted.

    Off unless PHYSIS_ALLOW_LOOPBACK_FETCH is set. It exists for one job: the
    development scripts serve test images from a throwaway local http server,
    and a presigned https URL is not something a laptop can produce.

    Deliberately narrower than "skip the checks". It permits loopback and
    nothing else, so the addresses an SSRF attempt actually wants stay blocked:
    169.254.169.254 is link-local, and a hospital's internal ranges are private.
    Never set on the Modal deployment.
    """
    return os.environ.get("PHYSIS_ALLOW_LOOPBACK_FETCH", "").strip().lower() in {"1", "true", "yes"}


def check_url(url: str, *, allowlist: tuple[str, ...] | None = None) -> None:
    """Run every gate that does not require a connection. Raises, or returns."""
    parts = urlsplit(url)
    if not parts.hostname:
        raise ImageHostRejected("image_url has no host")

    if loopback_allowed() and _is_loopback(parts.hostname):
        return
    if parts.scheme.lower() != "https":
        raise ImageHostRejected("image_url must be https")
    if not _host_allowed(parts.hostname, allowlist if allowlist is not None else allowed_hosts()):
        raise ImageHostRejected(f"host not allowed: {parts.hostname}")
    _addresses_are_public(parts.hostname)


def fetch_image(
    url: str,
    *,
    allowlist: tuple[str, ...] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = CONNECT_TIMEOUT_S,
) -> bytes:
    """A presigned URL, as bytes, or an exception naming which gate stopped it."""
    check_url(url, allowlist=allowlist)

    opener = urllib.request.build_opener(_NoRedirects)
    try:
        with opener.open(url, timeout=timeout) as response:
            # Content-Length is the server's claim and is checked first only
            # because it is free. The read below is what actually enforces the
            # ceiling, since a lying or absent header proves nothing.
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise ImageTooLarge(f"declared {declared} bytes, ceiling {max_bytes}")

            body = response.read(max_bytes + 1)
    except (ImageFetchError, urllib.error.HTTPError, urllib.error.URLError, OSError) as err:
        if isinstance(err, ImageFetchError):
            raise
        raise ImageUnreachable(str(err)) from err

    if len(body) > max_bytes:
        raise ImageTooLarge(f"body exceeds {max_bytes} bytes")
    if not body:
        raise ImageUnreachable("empty response")
    return body


def redact(url: str) -> str:
    """`https://host/path` with the query string dropped.

    A presigned URL carries its authorization in the query string, so logging
    one whole hands out a working credential to anyone who can read the log.
    The path is kept because it identifies the object.
    """
    parts = urlsplit(url)
    # hostname and port rather than netloc: netloc would carry any user:password
    # prefix, which is the kind of thing this function exists to drop.
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return f"{parts.scheme}://{host}{parts.path}"


__all__ = [
    "DEFAULT_MAX_BYTES",
    "ImageFetchError",
    "ImageHostRejected",
    "ImageTooLarge",
    "ImageUnreachable",
    "allowed_hosts",
    "check_url",
    "fetch_image",
    "loopback_allowed",
    "redact",
]

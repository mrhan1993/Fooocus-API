"""Helpers for safely fetching remote resources (SSRF protection)."""
import ipaddress
import socket
import urllib.parse

import requests

BLOCKED_NETWORKS = (
    ipaddress.ip_network('0.0.0.0/8'),
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('100.64.0.0/10'),
    ipaddress.ip_network('127.0.0.0/8'),
    ipaddress.ip_network('169.254.0.0/16'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('192.0.0.0/24'),
    ipaddress.ip_network('192.0.2.0/24'),
    ipaddress.ip_network('198.18.0.0/15'),
    ipaddress.ip_network('198.51.100.0/24'),
    ipaddress.ip_network('203.0.113.0/24'),
    ipaddress.ip_network('224.0.0.0/4'),
    ipaddress.ip_network('240.0.0.0/4'),
    ipaddress.ip_network('255.255.255.255/32'),
    ipaddress.ip_network('::/128'),
    ipaddress.ip_network('::1/128'),
    ipaddress.ip_network('::ffff:0:0/96'),
    ipaddress.ip_network('64:ff9b::/96'),
    ipaddress.ip_network('100::/64'),
    ipaddress.ip_network('2001:db8::/32'),
    ipaddress.ip_network('fc00::/7'),
    ipaddress.ip_network('fe80::/10'),
    ipaddress.ip_network('ff00::/8'),
)

_MAX_REDIRECTS = 5


def assert_url_allowed(url: str) -> None:
    """Raise ValueError if url points to a blocked (private/internal) host."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ('http', 'https'):
        raise ValueError(f"Unsupported URL scheme: {parts.scheme!r}")
    host = parts.hostname
    if not host:
        raise ValueError(f"Invalid URL, no host: {url}")
    port = parts.port or (443 if parts.scheme == 'https' else 80)
    try:
        addresses = socket.getaddrinfo(
            host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"Unable to resolve host: {host}") from e
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if any(ip in network for network in BLOCKED_NETWORKS):
            raise ValueError(f"URL host resolves to blocked address: {host} -> {ip}")


def _safe_request(method, url, timeout=10, headers=None, stream=False, json=None, data=None):
    """Perform an HTTP request, validating every hop against blocked addresses."""
    current_url = url
    for _ in range(_MAX_REDIRECTS + 1):
        assert_url_allowed(current_url)
        response = requests.request(
            method, current_url, timeout=timeout, headers=headers,
            stream=stream, json=json, data=data, allow_redirects=False)
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get('Location')
            response.close()
            if not location:
                break
            current_url = urllib.parse.urljoin(current_url, location)
            continue
        return response
    raise ValueError(f"Too many redirects for URL: {url}")


def safe_requests_get(url, timeout=10, headers=None, stream=False):
    """GET url with SSRF protection on the initial request and every redirect hop."""
    return _safe_request('GET', url, timeout=timeout, headers=headers, stream=stream)


def safe_requests_post(url, timeout=10, headers=None, json=None, data=None):
    """POST url with SSRF protection on the initial request and every redirect hop."""
    return _safe_request('POST', url, timeout=timeout, headers=headers, json=json, data=data)

"""Observe SDK allocation IDs without retrying creation requests."""
import re
import time

import httpx


class SandboxStartupTransport(httpx.AsyncBaseTransport):
    def __init__(self, base_url, on_created, *, transport=None, on_request=None):
        self.base_url = base_url.rstrip('/')
        self.on_created = on_created
        self.transport = transport or httpx.AsyncHTTPTransport()
        self.on_request = on_request

    def record_request(self, request, started, *, status=None, error=None):
        """Persist only lifecycle route templates, never URLs or SDK messages."""
        if self.on_request is None:
            return
        url = str(request.url).split('?', 1)[0]
        operation = 'other'
        for route, name in ((r'/sandboxes', 'sandboxes'),
                            (r'/sandboxes/[A-Za-z0-9_-]+', 'sandbox'),
                            (r'/sandboxes/[A-Za-z0-9_-]+/endpoints/\d+', 'endpoint')):
            if re.fullmatch(re.escape(self.base_url) + route, url):
                operation = name
                break
        self.on_request({'method': request.method, 'operation': operation, 'http_status': status,
                         'error_type': error, 'elapsed_sec': time.monotonic() - started, 'time': time.time()})

    async def handle_async_request(self, request):
        url = str(request.url).split('?', 1)[0]
        started = time.monotonic()
        try:
            response = await self.transport.handle_async_request(request)
        except BaseException as exc:
            self.record_request(request, started, error=type(exc).__name__)
            raise
        self.record_request(request, started, status=response.status_code)
        if request.method == 'POST' and url == self.base_url + '/sandboxes' and response.is_success:
            await response.aread()
            try:
                sandbox_id = response.json()['id']
                if isinstance(sandbox_id, str) and re.fullmatch(r'[A-Za-z0-9_-]+', sandbox_id):
                    self.on_created(sandbox_id)
            except (ValueError, KeyError, TypeError):
                pass  # Let the SDK reject malformed creation responses.
        return response

    async def aclose(self):
        await self.transport.aclose()

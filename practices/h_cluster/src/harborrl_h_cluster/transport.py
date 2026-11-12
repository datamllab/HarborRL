"""Bounded retries for H-platform endpoint discovery, never allocation POSTs."""
import asyncio
import re
import time

from harborrl.rollout.harbor_job.sandbox_transport import (
    SandboxStartupTransport as AllocationTransport,
)


class SandboxStartupTransport(AllocationTransport):
    def __init__(self, base_url, on_created, *, timeout, transport=None, retry_delay=2, on_request=None):
        super().__init__(base_url, on_created, transport=transport, on_request=on_request)
        self.deadline = time.monotonic() + timeout
        self.retry_delay = retry_delay

    async def handle_async_request(self, request):
        url = str(request.url).split('?', 1)[0]
        endpoint_get = request.method == 'GET' and re.fullmatch(
            re.escape(self.base_url) + r'/sandboxes/[A-Za-z0-9_-]+/endpoints/\d+', url)
        while True:
            response = await super().handle_async_request(request)
            if not endpoint_get or response.status_code not in (502, 503, 504):
                return response
            if self.deadline - time.monotonic() <= self.retry_delay:
                return response
            await response.aclose()
            await asyncio.sleep(self.retry_delay)

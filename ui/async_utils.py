from __future__ import annotations

import asyncio
import concurrent.futures


def run(coro):
    """Run async coro in an isolated worker thread — never touches Streamlit's anyio loop."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result(timeout=1800)  # 30-min max for long eval runs

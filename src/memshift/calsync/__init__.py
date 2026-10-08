"""Calendar official sync semantics (plan §4.2 "同步语义", §2.5 H_S, §6 E4).

The main results use ideal sync: every source event reaches the data layer once, at its own time.
E4 replaces the calendar source with a simulated server that follows the official incremental
sync contract (sync tokens, pagination, deletions as cancelled tombstones, HTTP 410 on expired
tokens, ETag `If-Match` conditional writes, read-after-write lag L) and a compliant client that
polls it; the data layer then sees the calendar only through that client.
"""

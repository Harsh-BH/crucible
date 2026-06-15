"""Async client for the Sentinel sandbox API + :class:`SentinelVerifier`.

Sentinel (https://github.com/Harsh-BH/Sentinel) executes a single source file in
a hardened nsjail/cgroups-v2 sandbox (no network). It is the **hardened**
execution backend on Crucible's weak->hardened axis -- the counterpart to the
deliberately weak :class:`verifier.backends.LocalPyVerifier`.

API (async, default base ``http://localhost:8080``, prefix ``/api/v1``):

* ``POST /api/v1/submissions`` with ``{"language","source_code","stdin",
  "time_limit_ms"?,"memory_limit_kb"?}`` -> ``202 {"job_id","status":"QUEUED"}``
* ``GET  /api/v1/submissions/{job_id}`` -> a job document with at least a
  ``status`` field and, on completion, execution metrics.

Terminal statuses: ``SUCCESS``, ``COMPILATION_ERROR``, ``RUNTIME_ERROR``,
``TIMEOUT``, ``MEMORY_LIMIT_EXCEEDED``, ``INTERNAL_ERROR``.

Field-name robustness
---------------------
The public docs pin the request body and the 202 response but do not freeze the
*result* field names. We therefore read result metrics defensively, preferring
the documented/spec names (``time_used_ms``, ``memory_used_kb``, ``exit_code``,
``stdout``, ``stderr``) and falling back to common aliases. If Sentinel renames a
field, only :func:`_pick` mappings change.

Sandbox-signal gap (documented)
-------------------------------
Sentinel does **not** today surface seccomp-violation or network-attempt
signals, so :class:`SentinelVerifier` leaves ``hack_flags.seccomp_violation`` and
``hack_flags.network_attempt`` ``False``. It *does* derive ``timed_out`` /
``oom_killed`` / ``resource_exhaustion`` from the terminal status and exit code.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from .smoke.checks import build_python_harness, parse_harness_output
from .types import ArtifactKind, VerifyResult, VerifySpec

__all__ = ["SentinelClient", "SentinelVerifier", "TERMINAL_STATUSES"]

#: Statuses past which a job will not change.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "SUCCESS",
        "COMPILATION_ERROR",
        "RUNTIME_ERROR",
        "TIMEOUT",
        "MEMORY_LIMIT_EXCEEDED",
        "INTERNAL_ERROR",
    }
)

_API_PREFIX = "/api/v1"
_MAX_TAIL = 4096  # cap captured stdout/stderr


def _tail(s: Any, n: int = _MAX_TAIL) -> str:
    if not s:
        return ""
    s = s if isinstance(s, str) else str(s)
    return s[-n:]


def _pick(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return ``d[k]`` for the first present, non-None ``k`` in ``keys``."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


class SentinelClient:
    """Thin async HTTP client for the Sentinel submissions API.

    Usable as an async context manager::

        async with SentinelClient() as c:
            job = await c.run("print(1)")

    A custom ``transport`` (e.g. ``httpx.MockTransport``) may be injected for
    tests; when given, ``base_url``/``timeout`` still apply.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 30.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        headers = {"Accept": "application/json"}
        if api_key:
            # Sentinel has no auth today; supported for when it is added.
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
            transport=transport,
        )

    async def __aenter__(self) -> SentinelClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def submit(
        self,
        source_code: str,
        language: str = "python",
        stdin: str = "",
        time_limit_ms: int | None = None,
        memory_limit_kb: int | None = None,
    ) -> str:
        """Submit ``source_code``; return the ``job_id``.

        ``POST {base}/api/v1/submissions``. Expects ``202`` with
        ``{"job_id","status":"QUEUED"}``. Raises ``httpx.HTTPStatusError`` on a
        non-202 response.
        """
        body: dict[str, Any] = {
            "language": language,
            "source_code": source_code,
            "stdin": stdin,
        }
        if time_limit_ms is not None:
            body["time_limit_ms"] = int(time_limit_ms)
        if memory_limit_kb is not None:
            body["memory_limit_kb"] = int(memory_limit_kb)

        resp = await self._client.post(f"{_API_PREFIX}/submissions", json=body)
        if resp.status_code != 202:
            raise httpx.HTTPStatusError(
                f"submit expected 202, got {resp.status_code}: {_tail(resp.text, 512)}",
                request=resp.request,
                response=resp,
            )
        data = resp.json()
        job_id = _pick(data, "job_id", "id", "jobId")
        if not job_id:
            raise ValueError(f"submit response missing job_id: {data!r}")
        return str(job_id)

    async def get(self, job_id: str) -> dict[str, Any]:
        """``GET {base}/api/v1/submissions/{job_id}`` -> the job document."""
        resp = await self._client.get(f"{_API_PREFIX}/submissions/{job_id}")
        resp.raise_for_status()
        return resp.json()

    async def run(
        self,
        source_code: str,
        language: str = "python",
        stdin: str = "",
        time_limit_ms: int | None = None,
        memory_limit_kb: int | None = None,
        *,
        poll_interval: float = 0.25,
        deadline_s: float = 60.0,
    ) -> dict[str, Any]:
        """Submit then poll until a terminal status; return the job document.

        Raises :class:`TimeoutError` if no terminal status is reached within
        ``deadline_s`` (this is the *client* deadline; distinct from the sandbox
        ``time_limit_ms``).
        """
        job_id = await self.submit(
            source_code,
            language=language,
            stdin=stdin,
            time_limit_ms=time_limit_ms,
            memory_limit_kb=memory_limit_kb,
        )
        loop = asyncio.get_event_loop()
        start = loop.time()
        while True:
            job = await self.get(job_id)
            status = str(_pick(job, "status", default="")).upper()
            if status in TERMINAL_STATUSES:
                return job
            if loop.time() - start > deadline_s:
                raise TimeoutError(
                    f"Sentinel job {job_id} did not finish within {deadline_s}s "
                    f"(last status={status!r})"
                )
            await asyncio.sleep(poll_interval)


class SentinelVerifier:
    """``Verifier`` backed by the hardened Sentinel sandbox (``name='sentinel'``).

    Routing:
      * :attr:`ArtifactKind.PYTHON` -- submit the artifact *directly* as
        ``source_code`` (the M2 path: route raw code execution through the
        hardened sandbox). ``build_ok`` is set from a clean run; ``smoke_ok``
        from exit code ``0``.
      * any other kind (Dockerfile, ...) -- build the deterministic harness via
        :func:`build_python_harness` and submit *that* as Python, so the
        untrusted check logic runs inside nsjail/cgroups. The harness's stdout
        JSON yields ``build_ok``/``smoke_ok``.

    Limits are mapped from ``spec.limits``: ``time_limit_ms = wall_s*1000`` and
    ``memory_limit_kb = mem_mb*1024``.

    NOTE: Sentinel does not expose seccomp/network-violation signals today, so
    ``hack_flags.seccomp_violation`` and ``hack_flags.network_attempt`` stay
    ``False`` (documented gap). ``timed_out``/``oom_killed``/
    ``resource_exhaustion`` *are* derived.
    """

    name = "sentinel"

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 30.0,
        *,
        client: SentinelClient | None = None,
        poll_interval: float = 0.25,
        deadline_s: float = 60.0,
    ) -> None:
        self._owns_client = client is None
        self._client = client or SentinelClient(
            base_url=base_url, api_key=api_key, timeout=timeout
        )
        self.poll_interval = poll_interval
        self.deadline_s = deadline_s

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def verify(self, artifact: str, spec: VerifySpec) -> VerifyResult:
        is_python = spec.kind == ArtifactKind.PYTHON
        source = artifact if is_python else build_python_harness(artifact, spec)

        limits = spec.limits
        time_limit_ms = int(limits.wall_s * 1000) if limits.wall_s else None
        memory_limit_kb = int(limits.mem_mb * 1024) if limits.mem_mb else None

        result = VerifyResult(backend=self.name)
        try:
            job = await self._client.run(
                source,
                language="python",
                time_limit_ms=time_limit_ms,
                memory_limit_kb=memory_limit_kb,
                poll_interval=self.poll_interval,
                deadline_s=self.deadline_s,
            )
        except TimeoutError as exc:
            # Client gave up waiting -- treat as a timeout/resource-exhaustion.
            result.status = "client-timeout"
            result.stderr_tail = _tail(repr(exc))
            result.hack_flags.timed_out = True
            result.hack_flags.resource_exhaustion = True
            return result
        except httpx.HTTPError as exc:
            result.status = "sentinel-error"
            result.stderr_tail = _tail(repr(exc))
            return result

        return self._map_job(job, is_python=is_python)

    # -- mapping ------------------------------------------------------------

    def _map_job(self, job: dict[str, Any], *, is_python: bool) -> VerifyResult:
        status = str(_pick(job, "status", default="")).upper()
        exit_code = _pick(job, "exit_code", "exitCode", "return_code", "code")
        try:
            exit_code = int(exit_code) if exit_code is not None else None
        except (TypeError, ValueError):
            exit_code = None

        stdout = _pick(job, "stdout", "output", "std_out", default="") or ""
        stderr = _pick(job, "stderr", "std_err", default="") or ""
        compile_out = _pick(job, "compile_output", "compiler_output", default="") or ""

        # time_used_ms -> wall_s ; memory_used_kb -> mem_mb
        time_used_ms = _pick(
            job, "time_used_ms", "execution_time_ms", "duration_ms", "time_ms",
            "runtime_ms", default=0,
        )
        memory_used_kb = _pick(
            job, "memory_used_kb", "memory_kb", "max_rss_kb", "memory", default=0,
        )

        result = VerifyResult(backend=self.name)
        result.status = status
        result.exit_code = exit_code
        result.stdout_tail = _tail(stdout)
        result.stderr_tail = _tail(stderr + ("\n" + compile_out if compile_out else ""))
        try:
            result.wall_s = float(time_used_ms) / 1000.0
        except (TypeError, ValueError):
            result.wall_s = 0.0
        try:
            result.mem_mb = float(memory_used_kb) / 1024.0
        except (TypeError, ValueError):
            result.mem_mb = 0.0
        result.raw = {"job": job}

        hf = result.hack_flags
        if status == "TIMEOUT":
            hf.timed_out = True
            hf.resource_exhaustion = True
            return result
        if status == "MEMORY_LIMIT_EXCEEDED" or exit_code == 137:
            hf.oom_killed = True
            hf.resource_exhaustion = True
            return result
        if status in ("RUNTIME_ERROR", "COMPILATION_ERROR", "INTERNAL_ERROR"):
            # build/smoke stay False; status already recorded.
            return result

        if status == "SUCCESS":
            clean = exit_code in (0, None)
            if is_python:
                # Raw code path: build_ok == "it ran", smoke_ok == exit 0.
                result.build_ok = clean
                result.smoke_ok = exit_code == 0
            else:
                parsed = parse_harness_output(stdout)
                if parsed is not None:
                    result.build_ok = bool(parsed.get("build_ok"))
                    result.smoke_ok = bool(parsed.get("smoke_ok"))
                    signals = parsed.get("signals") or {}
                    if isinstance(signals, dict):
                        result.raw["signals"] = signals
                        if signals.get("spec_gaming"):
                            hf.spec_gaming = True
                    if parsed.get("reasons"):
                        result.raw["reasons"] = parsed["reasons"]
                else:
                    # Ran clean but emitted no parseable contract line.
                    result.status = "no-harness-json"
        return result

"""Verifier backends + the :func:`get_verifier` factory.

Backends span Crucible's weak->hardened execution axis:

============   =============================================================
``name``       backend / where the check logic runs
============   =============================================================
``static``     :class:`StaticVerifier` -- ``check_dockerfile`` IN-PROCESS. No
               sandbox. Always available. The universal fallback / lower bound.
``local-py``   :class:`LocalPyVerifier` -- the harness run as a local
               ``python3`` subprocess under *generous* ``resource`` limits.
               The deliberately weak execution baseline (C3 study).
``local-docker`` :class:`LocalDockerVerifier` -- the GENUINE ``docker build`` +
               ``docker run`` + HTTP health probe.
``local-compose`` :class:`LocalComposeVerifier` -- GENUINE ``docker compose up
               --build`` + HTTP health probe (the eval verifier for
               :attr:`ArtifactKind.COMPOSE`).
``sentinel``   :class:`verifier.sentinel_client.SentinelVerifier` -- the same
               harness submitted to the hardened nsjail/cgroups sandbox.
============   =============================================================

Every backend sets ``result.backend = self.name`` and fills ``wall_s``,
``exit_code``, ``stdout_tail``, ``stderr_tail``, ``status`` and ``hack_flags``
where derivable, and leaves ``reward=None`` (the environment applies
:func:`verifier.reward.shape_reward`).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from .smoke.checks import (
    build_python_harness,
    check_artifact,
    parse_harness_output,
)
from .types import ArtifactKind, Verifier, VerifyResult, VerifySpec

__all__ = [
    "StaticVerifier",
    "LocalPyVerifier",
    "LocalDockerVerifier",
    "LocalComposeVerifier",
    "get_verifier",
]

_MAX_TAIL = 4096


def _tail(s: str | bytes | None, n: int = _MAX_TAIL) -> str:
    if not s:
        return ""
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    return s[-n:]


def _http_probe(url: str, expect_status: int, deadline_s: float) -> tuple[bool, str]:
    """Poll ``url`` until it returns ``expect_status`` or the deadline elapses.

    Shared by :class:`LocalDockerVerifier` and :class:`LocalComposeVerifier`
    (both serve an HTTP health endpoint after a genuine build+run). Returns
    ``(ok, detail)`` where ``detail`` is the last observed response/error.
    """
    end = time.monotonic() + deadline_s
    last = ""
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                code = resp.getcode()
                if code == expect_status:
                    return True, f"HTTP {code}"
                last = f"HTTP {code}"
        except urllib.error.HTTPError as e:
            if e.code == expect_status:
                return True, f"HTTP {e.code}"
            last = f"HTTP {e.code}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = repr(e)
        time.sleep(0.4)
    return False, last or "no response before deadline"


# ---------------------------------------------------------------------------
# StaticVerifier
# ---------------------------------------------------------------------------
class StaticVerifier:
    """In-process static analysis. ``name='static'``. No execution sandbox.

    The universal fallback / weak lower bound: runs :func:`check_artifact`
    (dispatching on ``spec.kind`` to :func:`check_dockerfile` /
    :func:`check_compose`) directly and maps the result. Never times out, never
    OOMs.
    """

    name = "static"

    async def verify(self, artifact: str, spec: VerifySpec) -> VerifyResult:
        t0 = time.monotonic()
        out = check_artifact(artifact, spec)
        result = VerifyResult(
            build_ok=bool(out["build_ok"]),
            smoke_ok=bool(out["smoke_ok"]),
            exit_code=0,
            status="ok",
            backend=self.name,
            wall_s=time.monotonic() - t0,
            raw={"signals": out["signals"], "reasons": out["reasons"]},
        )
        if out["signals"].get("spec_gaming"):
            result.hack_flags.spec_gaming = True
        return result


# ---------------------------------------------------------------------------
# LocalPyVerifier
# ---------------------------------------------------------------------------
def _apply_rlimits(mem_mb: int, cpu_s: int, pids: int) -> None:
    """Best-effort ``setrlimit`` preexec hook (Linux). Generous on purpose."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-Linux
        return
    try:
        cpu = max(1, int(cpu_s))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    except (ValueError, OSError):
        pass
    try:
        if mem_mb:
            nbytes = int(mem_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
    except (ValueError, OSError):
        pass
    try:
        if pids and hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (int(pids), int(pids)))
    except (ValueError, OSError):
        pass


class LocalPyVerifier:
    """Run the harness as a local ``python3`` subprocess. ``name='local-py'``.

    The deliberately weak/naive execution baseline for the C3 study: a plain
    subprocess under *generous* ``resource`` limits (``RLIMIT_CPU`` /
    ``RLIMIT_AS`` / ``RLIMIT_NPROC`` where available) and a wall-clock timeout.
    No filesystem/network isolation -- contrast with :class:`SentinelVerifier`.

    Mapping: wall-clock timeout -> ``timed_out`` + ``resource_exhaustion``;
    exit 137 / ``MemoryError`` in stderr -> ``oom_killed`` + ``resource_exhaustion``;
    other non-zero exits -> build/smoke ``False`` with ``status`` recorded.

    :attr:`ArtifactKind.PYTHON` artifacts run *directly* as the subprocess source
    (no harness) -- mirroring :class:`SentinelVerifier` -- so the C3 weak vs
    hardened comparison executes the same raw code. For that path ``build_ok``
    means "it ran" and ``smoke_ok`` means "exited 0".
    """

    name = "local-py"

    def __init__(
        self,
        *,
        time_limit_ms: int | None = None,
        mem_mb: int | None = None,
        python_exe: str | None = None,
    ) -> None:
        self.time_limit_ms = time_limit_ms
        self.mem_mb = mem_mb
        self.python_exe = python_exe or sys.executable or "python3"

    async def verify(self, artifact: str, spec: VerifySpec) -> VerifyResult:
        is_python = spec.kind == ArtifactKind.PYTHON
        # PYTHON: run the artifact itself (mirrors SentinelVerifier); any other
        # kind: wrap it in the deterministic harness.
        source = artifact if is_python else build_python_harness(artifact, spec)
        limits = spec.limits
        # Generous: prefer explicit override, else 4x the spec wall (this is the
        # weak baseline -- it should rarely kill legitimate work).
        wall_s = (
            self.time_limit_ms / 1000.0
            if self.time_limit_ms is not None
            else max(5.0, float(limits.wall_s) * 4.0)
        )
        mem_mb = self.mem_mb if self.mem_mb is not None else max(256, limits.mem_mb * 4)
        cpu_s = int(wall_s) + 2
        pids = max(limits.pids, 64)

        return await asyncio.to_thread(
            self._run_blocking, source, wall_s, mem_mb, cpu_s, pids, is_python
        )

    def _run_blocking(
        self,
        source: str,
        wall_s: float,
        mem_mb: int,
        cpu_s: int,
        pids: int,
        is_python: bool = False,
    ) -> VerifyResult:
        result = VerifyResult(backend=self.name)
        preexec = None
        if os.name == "posix":
            preexec = lambda: _apply_rlimits(mem_mb, cpu_s, pids)  # noqa: E731

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [self.python_exe, "-I", "-c", source],
                capture_output=True,
                text=True,
                timeout=wall_s,
                preexec_fn=preexec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            result.wall_s = time.monotonic() - t0
            result.status = "timeout"
            result.stdout_tail = _tail(exc.stdout)
            result.stderr_tail = _tail(exc.stderr)
            result.hack_flags.timed_out = True
            result.hack_flags.resource_exhaustion = True
            return result
        except (OSError, ValueError) as exc:
            result.wall_s = time.monotonic() - t0
            result.status = "spawn-error"
            result.stderr_tail = _tail(repr(exc))
            return result

        result.wall_s = time.monotonic() - t0
        result.exit_code = proc.returncode
        result.stdout_tail = _tail(proc.stdout)
        result.stderr_tail = _tail(proc.stderr)

        # OOM signals: SIGKILL(137) or a MemoryError surfaced in stderr.
        if proc.returncode == 137 or "MemoryError" in (proc.stderr or ""):
            result.status = "oom"
            result.hack_flags.oom_killed = True
            result.hack_flags.resource_exhaustion = True
            return result
        # Killed by signal (negative return code under POSIX).
        if proc.returncode is not None and proc.returncode < 0:
            result.status = f"signal-{-proc.returncode}"
            result.hack_flags.resource_exhaustion = True
            return result

        if is_python:
            # Raw code path: no harness JSON. build_ok == "it ran", smoke_ok ==
            # exit 0 (mirrors SentinelVerifier). No spec_gaming signal possible.
            result.build_ok = True
            result.smoke_ok = proc.returncode == 0
            result.status = "ok" if proc.returncode == 0 else "nonzero-exit"
            return result

        parsed = parse_harness_output(proc.stdout or "")
        if proc.returncode == 0 and parsed is not None:
            result.status = "ok"
            result.build_ok = bool(parsed.get("build_ok"))
            result.smoke_ok = bool(parsed.get("smoke_ok"))
            signals = parsed.get("signals") or {}
            if isinstance(signals, dict):
                result.raw["signals"] = signals
                if signals.get("spec_gaming"):
                    result.hack_flags.spec_gaming = True
            if parsed.get("reasons"):
                result.raw["reasons"] = parsed["reasons"]
        else:
            result.status = "nonzero-exit" if proc.returncode != 0 else "no-harness-json"
        return result


# ---------------------------------------------------------------------------
# LocalDockerVerifier
# ---------------------------------------------------------------------------
class LocalDockerVerifier:
    """Genuine ``docker build`` + ``docker run`` + HTTP smoke probe.

    ``name='local-docker'``. Writes the artifact to a temp build context, builds
    with ``--memory``/``--cpus`` and a build timeout, runs the image detached
    with the port published, polls ``http://localhost:<port><health_path>`` for
    ``expect_status`` (default 200), then tears everything down. ``build_ok`` is
    the build outcome; ``smoke_ok`` is the probe outcome.

    If the ``docker`` CLI is unavailable, :meth:`verify` returns
    ``VerifyResult(build_ok=False, status='docker-unavailable')`` (it never
    raises). All blocking work runs via :func:`asyncio.to_thread`.

    The build/run steps are factored into overridable hooks
    (:meth:`_docker_build`, :meth:`_docker_run`, :meth:`_probe`) so result
    mapping can be unit-tested without a real daemon.
    """

    name = "local-docker"

    def __init__(
        self,
        *,
        time_limit_ms: int | None = None,
        mem_mb: int | None = None,
        docker_exe: str | None = None,
        remove_image: bool = True,
    ) -> None:
        self.time_limit_ms = time_limit_ms
        self.mem_mb = mem_mb
        self.docker_exe = docker_exe or "docker"
        self.remove_image = remove_image

    def _docker_available(self) -> bool:
        return shutil.which(self.docker_exe) is not None

    async def verify(self, artifact: str, spec: VerifySpec) -> VerifyResult:
        if not self._docker_available():
            return VerifyResult(
                backend=self.name,
                status="docker-unavailable",
                build_ok=False,
                stderr_tail="docker CLI not found on PATH",
            )
        return await asyncio.to_thread(self._run_blocking, artifact, spec)

    # -- overridable hooks (for testing) -----------------------------------
    def _docker_build(
        self, context_dir: str, tag: str, mem_mb: int, cpus: float, timeout_s: float
    ) -> subprocess.CompletedProcess:
        cmd = [
            self.docker_exe, "build", "--rm", "-t", tag,
            f"--memory={mem_mb}m", "-f", os.path.join(context_dir, "Dockerfile"),
            context_dir,
        ]
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )

    def _docker_run(
        self, tag: str, host_port: int, container_port: int, mem_mb: int, cpus: float
    ) -> subprocess.CompletedProcess:
        cmd = [
            self.docker_exe, "run", "-d", "--rm",
            f"--memory={mem_mb}m", f"--cpus={cpus}",
            "-p", f"{host_port}:{container_port}", tag,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)

    def _probe(self, url: str, expect_status: int, deadline_s: float) -> tuple[bool, str]:
        """Poll ``url`` until it returns ``expect_status`` or the deadline."""
        return _http_probe(url, expect_status, deadline_s)

    def _docker_stop(self, container_id: str) -> None:
        try:
            subprocess.run(
                [self.docker_exe, "stop", container_id],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    def _docker_rmi(self, tag: str) -> None:
        if not self.remove_image:
            return
        try:
            subprocess.run(
                [self.docker_exe, "rmi", "-f", tag],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    # -- orchestration ------------------------------------------------------
    def _run_blocking(self, artifact: str, spec: VerifySpec) -> VerifyResult:
        result = VerifyResult(backend=self.name)
        limits = spec.limits
        smoke = spec.smoke or {}
        mem_mb = self.mem_mb if self.mem_mb is not None else limits.mem_mb
        cpus = float(limits.cpus or 1.0)
        build_timeout = (
            self.time_limit_ms / 1000.0
            if self.time_limit_ms is not None
            else max(60.0, float(limits.wall_s) * 4.0)
        )
        container_port = smoke.get("port")
        try:
            container_port = int(container_port) if container_port is not None else 8000
        except (TypeError, ValueError):
            container_port = 8000
        health_path = smoke.get("health_path", "/")
        if not str(health_path).startswith("/"):
            health_path = "/" + str(health_path)
        expect_status = int(smoke.get("expect_status", 200))

        tag = f"crucible-verify:{os.getpid()}-{int(time.time() * 1000) & 0xffffff}"
        t0 = time.monotonic()
        tmpdir = tempfile.mkdtemp(prefix="crucible-docker-")
        container_id = ""
        try:
            with open(os.path.join(tmpdir, "Dockerfile"), "w", encoding="utf-8") as fh:
                fh.write(artifact or "")

            # Scaffold/app files the Dockerfile COPYs (requirements.txt, app/…).
            # ``context_files`` maps POSIX-relative paths -> contents; each must
            # stay inside the build context (guard against ``..`` traversal).
            for relpath, content in (smoke.get("context_files") or {}).items():
                dest = os.path.normpath(os.path.join(tmpdir, relpath))
                if dest != tmpdir and not dest.startswith(tmpdir + os.sep):
                    raise ValueError(f"context_files path escapes build context: {relpath!r}")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "w", encoding="utf-8") as fh:
                    fh.write(content if isinstance(content, str) else str(content))

            # --- build --------------------------------------------------
            try:
                build = self._docker_build(tmpdir, tag, mem_mb, cpus, build_timeout)
            except subprocess.TimeoutExpired as exc:
                result.status = "build-timeout"
                result.stdout_tail = _tail(exc.stdout)
                result.stderr_tail = _tail(exc.stderr)
                result.hack_flags.timed_out = True
                result.hack_flags.resource_exhaustion = True
                result.wall_s = time.monotonic() - t0
                return result

            result.exit_code = build.returncode
            result.stdout_tail = _tail(build.stdout)
            build_err = build.stderr or ""
            result.stderr_tail = _tail(build_err)
            if build.returncode != 0:
                result.status = "build-failed"
                if _looks_like_oom(build_err) or build.returncode == 137:
                    result.hack_flags.oom_killed = True
                    result.hack_flags.resource_exhaustion = True
                result.wall_s = time.monotonic() - t0
                return result

            result.build_ok = True
            result.status = "built"

            # --- run ----------------------------------------------------
            host_port = container_port
            try:
                run = self._docker_run(tag, host_port, container_port, mem_mb, cpus)
            except subprocess.TimeoutExpired as exc:
                result.status = "run-timeout"
                result.stderr_tail = _tail(exc.stderr)
                result.wall_s = time.monotonic() - t0
                return result
            if run.returncode != 0:
                result.status = "run-failed"
                result.stderr_tail = _tail(run.stderr)
                result.wall_s = time.monotonic() - t0
                return result
            container_id = (run.stdout or "").strip().splitlines()[0] if run.stdout else ""

            # --- probe --------------------------------------------------
            probe_deadline = max(5.0, float(limits.wall_s))
            url = f"http://localhost:{host_port}{health_path}"
            ok, detail = self._probe(url, expect_status, probe_deadline)
            result.smoke_ok = ok
            result.status = "smoke-ok" if ok else "smoke-failed"
            result.stderr_tail = _tail(result.stderr_tail + "\nprobe: " + detail)
            result.wall_s = time.monotonic() - t0
            return result
        finally:
            if container_id:
                self._docker_stop(container_id)
            self._docker_rmi(tag)
            shutil.rmtree(tmpdir, ignore_errors=True)


def _looks_like_oom(text: str) -> bool:
    t = (text or "").lower()
    return "out of memory" in t or "oom" in t or "killed" in t or "cannot allocate" in t


# ---------------------------------------------------------------------------
# LocalComposeVerifier
# ---------------------------------------------------------------------------
class LocalComposeVerifier:
    """Genuine ``docker compose up --build`` + HTTP smoke probe.

    ``name='local-compose'``. The real (non-static) eval verifier for
    :attr:`ArtifactKind.COMPOSE`: writes the artifact as ``docker-compose.yml``
    plus its build context to a temp dir, brings the stack up detached
    (``docker compose up -d --build``) under a unique project name, polls
    ``http://localhost:<port><health_path>`` for ``expect_status`` (default
    200), then tears everything down (``docker compose down -v``).
    ``build_ok`` is the up/build outcome; ``smoke_ok`` is the probe outcome.

    If the ``docker`` CLI is unavailable, :meth:`verify` returns
    ``VerifyResult(build_ok=False, status='docker-unavailable')`` (it never
    raises). All blocking work runs via :func:`asyncio.to_thread`.

    The up/down steps are factored into overridable hooks (:meth:`_compose_up`,
    :meth:`_probe`, :meth:`_compose_down`) so result mapping can be unit-tested
    without a real daemon.
    """

    name = "local-compose"

    def __init__(
        self,
        *,
        time_limit_ms: int | None = None,
        mem_mb: int | None = None,
        docker_exe: str = "docker",
        remove_volumes: bool = True,
    ) -> None:
        self.time_limit_ms = time_limit_ms
        self.mem_mb = mem_mb
        self.docker_exe = docker_exe
        self.remove_volumes = remove_volumes

    def _compose_available(self) -> bool:
        # ``compose`` is a ``docker`` subcommand; the CLI on PATH is enough.
        return shutil.which(self.docker_exe) is not None

    async def verify(self, artifact: str, spec: VerifySpec) -> VerifyResult:
        if not self._compose_available():
            return VerifyResult(
                backend=self.name,
                status="docker-unavailable",
                build_ok=False,
                stderr_tail="docker CLI not found on PATH",
            )
        return await asyncio.to_thread(self._run_blocking, artifact, spec)

    # -- overridable hooks (for testing) -----------------------------------
    def _compose_up(
        self, context_dir: str, project: str, mem_mb: int, timeout_s: float
    ) -> subprocess.CompletedProcess:
        cmd = [
            self.docker_exe, "compose", "-p", project,
            "-f", os.path.join(context_dir, "docker-compose.yml"),
            "up", "-d", "--build",
        ]
        return subprocess.run(
            cmd, cwd=context_dir, capture_output=True, text=True,
            timeout=timeout_s, check=False,
        )

    def _probe(self, url: str, expect_status: int, deadline_s: float) -> tuple[bool, str]:
        """Poll ``url`` until it returns ``expect_status`` or the deadline."""
        return _http_probe(url, expect_status, deadline_s)

    def _compose_down(self, context_dir: str, project: str) -> None:
        cmd = [
            self.docker_exe, "compose", "-p", project,
            "-f", os.path.join(context_dir, "docker-compose.yml"),
            "down",
        ]
        if self.remove_volumes:
            cmd.append("-v")
        try:
            subprocess.run(
                cmd, cwd=context_dir, capture_output=True, text=True,
                timeout=60, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    # -- orchestration ------------------------------------------------------
    def _run_blocking(self, artifact: str, spec: VerifySpec) -> VerifyResult:
        result = VerifyResult(backend=self.name)
        limits = spec.limits
        smoke = spec.smoke or {}
        mem_mb = self.mem_mb if self.mem_mb is not None else limits.mem_mb
        # Compose build can pull a base image + a db image: give it room.
        up_timeout = (
            self.time_limit_ms / 1000.0
            if self.time_limit_ms is not None
            else max(120.0, float(limits.wall_s) * 8.0)
        )
        port = smoke.get("port")
        try:
            port = int(port) if port is not None else 8000
        except (TypeError, ValueError):
            port = 8000
        health_path = smoke.get("health_path", "/")
        if not str(health_path).startswith("/"):
            health_path = "/" + str(health_path)
        expect_status = int(smoke.get("expect_status", 200))

        # Unique project name so concurrent / leftover runs never collide.
        project = f"crucible-compose-{os.getpid()}-{int(time.time() * 1000) & 0xffffff}"
        t0 = time.monotonic()
        tmpdir = tempfile.mkdtemp(prefix="crucible-compose-")
        try:
            with open(os.path.join(tmpdir, "docker-compose.yml"), "w", encoding="utf-8") as fh:
                fh.write(artifact or "")

            # Build context the compose services reference (Dockerfile,
            # requirements.txt, app/…). ``context_files`` maps POSIX-relative
            # paths -> contents; each must stay inside the temp dir (guard
            # against ``..`` traversal).
            for relpath, content in (smoke.get("context_files") or {}).items():
                dest = os.path.normpath(os.path.join(tmpdir, relpath))
                if dest != tmpdir and not dest.startswith(tmpdir + os.sep):
                    raise ValueError(f"context_files path escapes build context: {relpath!r}")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "w", encoding="utf-8") as fh:
                    fh.write(content if isinstance(content, str) else str(content))

            # --- build + up ---------------------------------------------
            try:
                up = self._compose_up(tmpdir, project, mem_mb, up_timeout)
            except subprocess.TimeoutExpired as exc:
                result.status = "compose-timeout"
                result.stdout_tail = _tail(exc.stdout)
                result.stderr_tail = _tail(exc.stderr)
                result.hack_flags.timed_out = True
                result.hack_flags.resource_exhaustion = True
                result.wall_s = time.monotonic() - t0
                return result

            result.exit_code = up.returncode
            result.stdout_tail = _tail(up.stdout)
            up_err = up.stderr or ""
            result.stderr_tail = _tail(up_err)
            if up.returncode != 0:
                result.status = "compose-up-failed"
                if _looks_like_oom(up_err) or up.returncode == 137:
                    result.hack_flags.oom_killed = True
                    result.hack_flags.resource_exhaustion = True
                result.wall_s = time.monotonic() - t0
                return result

            result.build_ok = True
            result.status = "up"

            # --- probe --------------------------------------------------
            probe_deadline = max(5.0, float(limits.wall_s))
            url = f"http://localhost:{port}{health_path}"
            ok, detail = self._probe(url, expect_status, probe_deadline)
            result.smoke_ok = ok
            result.status = "smoke-ok" if ok else "smoke-failed"
            result.stderr_tail = _tail(result.stderr_tail + "\nprobe: " + detail)
            result.wall_s = time.monotonic() - t0
            return result
        finally:
            self._compose_down(tmpdir, project)
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_verifier(
    name: str = "static",
    *,
    base_url: str | None = None,
    time_limit_ms: int | None = None,
    mem_mb: int | None = None,
    **kwargs: object,
) -> Verifier:
    """Construct a verifier backend by ``name``.

    ``name`` in ``{"static", "local-py", "local-docker", "local-compose",
    "sentinel"}``. Extra ``kwargs`` are forwarded to the backend constructor.
    ``base_url`` is used by ``sentinel``. Raises :class:`ValueError` on an
    unknown name.

    ``verifier.sentinel_client`` is imported lazily so ``import verifier`` stays
    cheap and never requires a running Sentinel.
    """
    if name == "static":
        return StaticVerifier()
    if name == "local-py":
        return LocalPyVerifier(
            time_limit_ms=time_limit_ms, mem_mb=mem_mb, **kwargs  # type: ignore[arg-type]
        )
    if name == "local-docker":
        return LocalDockerVerifier(
            time_limit_ms=time_limit_ms, mem_mb=mem_mb, **kwargs  # type: ignore[arg-type]
        )
    if name == "local-compose":
        return LocalComposeVerifier(
            time_limit_ms=time_limit_ms, mem_mb=mem_mb, **kwargs  # type: ignore[arg-type]
        )
    if name == "sentinel":
        from .sentinel_client import SentinelVerifier  # lazy

        kw: dict[str, object] = dict(kwargs)
        if base_url is not None:
            kw["base_url"] = base_url
        return SentinelVerifier(**kw)  # type: ignore[arg-type]
    raise ValueError(
        f"unknown verifier name {name!r}; expected one of "
        "'static', 'local-py', 'local-docker', 'local-compose', 'sentinel'"
    )

"""The common result envelope (SPEC §5): evidence over assertion.

Every tool returns this shape so an agent never mistakes "no finding" for
"fully verified", and success is proven by artifacts/frames rather than prose.
"""

from __future__ import annotations

from typing import Any


def result(
    ok: bool,
    payload: dict[str, Any] | None = None,
    *,
    artifacts: list[str] | None = None,
    frames: list[str] | None = None,
    timing: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
    verdict: dict[str, Any] | None = None,
    obligations: list[dict[str, Any]] | None = None,
    limitations: list[str] | None = None,
    untrusted: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    env: dict[str, Any] = {"ok": ok, "result": payload or {}}
    if error is not None:
        env["error"] = error
    env["evidence"] = {
        "artifacts": artifacts or [],
        "frames": frames or [],
        "timing": timing or {},
    }
    if coverage is not None:
        env["coverage"] = coverage
    if verdict is not None:
        env["verdict"] = verdict
    if obligations is not None:
        env["obligations"] = obligations
    env["limitations"] = limitations or []
    if untrusted:
        env["provenance"] = {"untrusted_content": untrusted}
    return env


def ok(payload: dict[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
    return result(True, payload, **kw)


def err(message: str, payload: dict[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
    return result(False, payload, error=message, **kw)

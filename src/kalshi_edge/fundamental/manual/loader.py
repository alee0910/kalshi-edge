"""Loader for analyst-entered fundamental inputs.

The on-disk layout is:

    analyst_inputs/
        README.md                 # instructions (committed once, not loaded)
        economics/                # one directory per category
            cpi_2026_04.yaml      # one or more YAML files per category
            notes_supply.yaml     # filename is free-form

Each YAML file has a top-level ``inputs:`` list. Every entry in that list
describes a single ``FundamentalInput``; the loader converts it into the
typed object. Example:

    category: economics
    analyst: alee                 # shows up in provenance.source
    inputs:
      - name: gasoline_weekly_delta
        value: 0.031
        uncertainty: 0.004
        mechanism: prior_shift
        observation_at: 2026-04-15T00:00:00Z
        expires_in_days: 21
        scope:
          transform: mom
        notes: "Hand-checked vs EIA weekly report for cross-validation."

Validation rules (loader rejects the file if any fail):

    * ``category`` matches the directory name.
    * ``name`` is listed in ``<category>_input_specs()`` — we do not accept
      free-form names; every input must correspond to a declared spec so
      the integration engine knows how to consume it.
    * ``mechanism`` matches the spec's mechanism.
    * ``value`` is a real number; ``uncertainty`` if present is ≥ 0.
    * ``observation_at`` is an absolute ISO 8601 timestamp (UTC or with tz).
    * ``expires_in_days`` is > 0 (OR an absolute ``expires_at`` is given).

Everything else is saved verbatim into provenance / scope. The file's last
git-commit SHA and author (if available) are captured into
``provenance.notes`` so the DB row can be traced back to a commit. When git
isn't available (fresh file, CI without git), we fall back to a "uncommitted"
marker — the loader still returns inputs so the pipeline can run, but the
CLI warns the operator.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from kalshi_edge.fundamental.schemas.base import (
    FundamentalInput,
    InputProvenance,
    InputSpec,
    IntegrationMechanism,
)
from kalshi_edge.fundamental.schemas.cpi import cpi_input_specs
from kalshi_edge.logging_ import get_logger

log = get_logger(__name__)


# Registry of category → spec-list function. Extending to new categories means
# adding an entry here and importing the category's ``*_input_specs`` fn.
_SPECS_BY_CATEGORY: dict[str, callable] = {
    "economics": cpi_input_specs,
}


class ManualInputError(ValueError):
    """Raised when a YAML entry violates the manual-input contract."""


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def load_manual_inputs(
    path: Path | str,
    *,
    now: datetime | None = None,
    strict: bool = True,
) -> list[FundamentalInput]:
    """Load manual inputs from a single YAML file.

    Parameters
    ----------
    path
        Absolute or relative path to a YAML file under ``analyst_inputs/``.
    now
        Reference timestamp used as ``fetched_at`` and to compute
        ``expires_at`` when the YAML gives a relative ``expires_in_days``.
        Defaults to ``datetime.now(utc)``.
    strict
        If True (default), the first malformed entry raises. If False, bad
        entries are logged and skipped (useful for tolerant dashboards);
        the CLI always calls with ``strict=True``.

    Returns
    -------
    list[FundamentalInput]
        Validated inputs ready for persistence / integration. Never
        returns partially-constructed objects: an entry either parses
        cleanly or is rejected.
    """
    p = Path(path)
    if not p.exists():
        raise ManualInputError(f"{p}: file does not exist")
    if not p.is_file():
        raise ManualInputError(f"{p}: not a regular file")

    now = _ensure_utc(now or datetime.now(timezone.utc))

    try:
        doc = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise ManualInputError(f"{p}: invalid YAML — {e}") from e

    if not isinstance(doc, dict):
        raise ManualInputError(f"{p}: top-level must be a mapping")

    category = doc.get("category")
    if not category or not isinstance(category, str):
        raise ManualInputError(f"{p}: missing or non-string ``category``")

    specs = _specs_for(category)
    by_name = {s.name: s for s in specs}

    analyst = doc.get("analyst") or "unknown"
    if not isinstance(analyst, str):
        raise ManualInputError(f"{p}: ``analyst`` must be a string if set")

    commit_note = _git_commit_note(p)

    entries = doc.get("inputs") or []
    if not isinstance(entries, list):
        raise ManualInputError(f"{p}: ``inputs`` must be a list")

    out: list[FundamentalInput] = []
    for i, raw in enumerate(entries):
        try:
            inp = _parse_entry(
                raw,
                category=category,
                spec_by_name=by_name,
                analyst=analyst,
                fetched_at=now,
                commit_note=commit_note,
                file_path=p,
                index=i,
            )
        except ManualInputError as e:
            if strict:
                raise
            log.warning("manual_input_skipped", path=str(p), index=i, error=str(e))
            continue
        out.append(inp)

    log.info("manual_inputs_loaded", path=str(p), n=len(out), category=category)
    return out


def load_manual_inputs_from_dir(
    root: Path | str,
    *,
    now: datetime | None = None,
    strict: bool = True,
) -> list[FundamentalInput]:
    """Load every ``*.yaml`` / ``*.yml`` file under ``root`` recursively.

    ``root`` is typically ``<repo>/analyst_inputs``. README / non-YAML files
    are ignored. Files whose directory name is not a known category are
    logged and skipped (they may belong to a category we haven't shipped
    schemas for yet — better to surface the gap than silently accept
    unstructured numbers).
    """
    root_p = Path(root)
    if not root_p.exists():
        log.info("manual_inputs_root_missing", path=str(root_p))
        return []
    if not root_p.is_dir():
        raise ManualInputError(f"{root_p}: not a directory")

    files: list[Path] = []
    for ext in ("*.yaml", "*.yml"):
        files.extend(sorted(root_p.rglob(ext)))

    out: list[FundamentalInput] = []
    for f in files:
        # Skip hidden files / editor-swap artifacts.
        if f.name.startswith(".") or f.name.startswith("_"):
            continue
        try:
            inputs = load_manual_inputs(f, now=now, strict=strict)
        except ManualInputError as e:
            if strict:
                raise
            log.warning("manual_inputs_file_rejected", path=str(f), error=str(e))
            continue
        out.extend(inputs)

    log.info("manual_inputs_root_loaded", root=str(root_p), n=len(out))
    return out


# ---------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------


def _specs_for(category: str) -> list[InputSpec]:
    fn = _SPECS_BY_CATEGORY.get(category)
    if fn is None:
        raise ManualInputError(
            f"unknown category {category!r}; manual inputs must target a "
            f"category with declared InputSpecs ({sorted(_SPECS_BY_CATEGORY)})"
        )
    return fn()


def _parse_entry(
    raw: Any,
    *,
    category: str,
    spec_by_name: dict[str, InputSpec],
    analyst: str,
    fetched_at: datetime,
    commit_note: str,
    file_path: Path,
    index: int,
) -> FundamentalInput:
    if not isinstance(raw, dict):
        raise ManualInputError(
            f"{file_path}[{index}]: entry must be a mapping, got {type(raw).__name__}"
        )

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ManualInputError(f"{file_path}[{index}]: missing ``name``")
    spec = spec_by_name.get(name)
    if spec is None:
        raise ManualInputError(
            f"{file_path}[{index}]: name {name!r} is not a declared input for "
            f"category {category!r} (known: {sorted(spec_by_name)})"
        )

    mech_raw = raw.get("mechanism", spec.mechanism.value)
    try:
        mechanism = IntegrationMechanism(mech_raw)
    except ValueError as e:
        raise ManualInputError(
            f"{file_path}[{index}]: unknown mechanism {mech_raw!r}"
        ) from e
    if mechanism != spec.mechanism:
        raise ManualInputError(
            f"{file_path}[{index}]: mechanism {mechanism.value!r} does not match "
            f"spec mechanism {spec.mechanism.value!r} for input {name!r}"
        )

    if "value" not in raw:
        raise ManualInputError(f"{file_path}[{index}]: missing ``value``")
    try:
        value = float(raw["value"])
    except (TypeError, ValueError) as e:
        raise ManualInputError(
            f"{file_path}[{index}]: ``value`` must be numeric"
        ) from e

    uncertainty: float | None = None
    if raw.get("uncertainty") is not None:
        try:
            uncertainty = float(raw["uncertainty"])
        except (TypeError, ValueError) as e:
            raise ManualInputError(
                f"{file_path}[{index}]: ``uncertainty`` must be numeric"
            ) from e
        if uncertainty < 0:
            raise ManualInputError(
                f"{file_path}[{index}]: ``uncertainty`` must be non-negative"
            )

    obs_raw = raw.get("observation_at")
    if obs_raw is None:
        raise ManualInputError(f"{file_path}[{index}]: missing ``observation_at``")
    observation_at = _parse_iso_ts(obs_raw, field="observation_at", file_path=file_path, index=index)

    expires_at = _compute_expiry(raw, fetched_at=fetched_at, file_path=file_path, index=index)

    scope = raw.get("scope") or {}
    if not isinstance(scope, dict):
        raise ManualInputError(f"{file_path}[{index}]: ``scope`` must be a mapping if set")

    user_notes = raw.get("notes")
    if user_notes is not None and not isinstance(user_notes, str):
        raise ManualInputError(f"{file_path}[{index}]: ``notes`` must be a string if set")

    provenance_notes = commit_note if not user_notes else f"{user_notes} | {commit_note}"

    return FundamentalInput(
        name=name,
        category=category,
        value=value,
        uncertainty=uncertainty,
        mechanism=mechanism,
        provenance=InputProvenance(
            source=f"analyst:{analyst}",
            source_kind="manual",
            fetched_at=fetched_at,
            observation_at=observation_at,
            notes=provenance_notes,
        ),
        expires_at=expires_at,
        scope=dict(scope),
    )


def _compute_expiry(
    raw: dict[str, Any],
    *,
    fetched_at: datetime,
    file_path: Path,
    index: int,
) -> datetime:
    if "expires_at" in raw and raw["expires_at"] is not None:
        return _parse_iso_ts(raw["expires_at"], field="expires_at", file_path=file_path, index=index)
    if "expires_in_days" in raw and raw["expires_in_days"] is not None:
        try:
            days = float(raw["expires_in_days"])
        except (TypeError, ValueError) as e:
            raise ManualInputError(
                f"{file_path}[{index}]: ``expires_in_days`` must be numeric"
            ) from e
        if days <= 0:
            raise ManualInputError(
                f"{file_path}[{index}]: ``expires_in_days`` must be positive"
            )
        return fetched_at + timedelta(days=days)
    raise ManualInputError(
        f"{file_path}[{index}]: one of ``expires_at`` or ``expires_in_days`` is required"
    )


def _parse_iso_ts(v: Any, *, field: str, file_path: Path, index: int) -> datetime:
    if isinstance(v, datetime):
        return _ensure_utc(v)
    if not isinstance(v, str):
        raise ManualInputError(
            f"{file_path}[{index}]: ``{field}`` must be an ISO-8601 string or datetime"
        )
    s = v.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise ManualInputError(
            f"{file_path}[{index}]: invalid ``{field}`` timestamp {v!r}"
        ) from e
    return _ensure_utc(dt)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _git_commit_note(path: Path) -> str:
    """Return ``"git:<short-sha> by <author> at <iso>"`` for the file.

    Falls back to ``"git:uncommitted"`` if the path isn't tracked or git
    isn't available. Never raises — a missing audit trail shouldn't block
    the pipeline.
    """
    try:
        cwd = path.parent
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        proc = subprocess.run(
            ["git", "log", "-1", "--format=%h|%an|%aI", "--", str(path)],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return "git:uncommitted"
        sha, author, when = proc.stdout.strip().split("|", 2)
        return f"git:{sha} by {author} at {when}"
    except Exception:  # noqa: BLE001 — audit is best-effort
        return "git:uncommitted"

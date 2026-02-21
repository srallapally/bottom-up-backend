"""services/cloud_run_jobs.py

Minimal Cloud Run Jobs trigger helper.

Used by:
- POST /api/sessions/<id>/process (routes/upload.py)
- POST /api/sessions/<id>/mine    (routes/mining.py)

Assumptions:
- Your Cloud Run Jobs are already created.
- The job container reads SESSION_ID (+ optional OWNER_SUB) from env.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import google.auth  # type: ignore
from google.auth.transport.requests import Request  # type: ignore

try:
    import requests  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("requests is required for Cloud Run Jobs triggering") from e


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"{name} is required")
    return val


def _bearer_token() -> str:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    if not getattr(creds, "token", None):
        raise RuntimeError("Failed to obtain access token")
    return creds.token


def _run_job(job_name: str, session_id: str, owner_sub: str | None) -> Dict[str, Any]:
    project = _require_env("GCP_PROJECT")
    region = _require_env("GCP_REGION")

    url = (
        f"https://{region}-run.googleapis.com/apis/run.googleapis.com/v1/"
        f"namespaces/{project}/jobs/{job_name}:run"
    )

    env = [{"name": "SESSION_ID", "value": session_id}]
    if owner_sub:
        env.append({"name": "OWNER_SUB", "value": owner_sub})

    payload = {
        "overrides": {
            "containerOverrides": [
                {
                    "env": env,
                }
            ]
        }
    }

    token = _bearer_token()
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    if resp.status_code >= 400:
        raise RuntimeError(f"Cloud Run Jobs run failed: HTTP {resp.status_code} body={resp.text}")

    return resp.json() if resp.content else {}


def run_process_job(session_id: str, owner_sub: str | None = None) -> Dict[str, Any]:
    job_name = _require_env("CLOUD_RUN_PROCESS_JOB")
    return _run_job(job_name=job_name, session_id=session_id, owner_sub=owner_sub)


def run_mine_job(session_id: str, owner_sub: str | None = None) -> Dict[str, Any]:
    job_name = _require_env("CLOUD_RUN_MINE_JOB")
    return _run_job(job_name=job_name, session_id=session_id, owner_sub=owner_sub)

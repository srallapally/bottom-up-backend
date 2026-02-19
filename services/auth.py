from __future__ import annotations

import os
import logging
from functools import wraps
from flask import request, jsonify, g

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

logger = logging.getLogger(__name__)


def _get_bearer_token() -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return auth[len("Bearer "):].strip() or None


def require_auth(fn):
    """
    Verifies Google ID token from Authorization: Bearer <jwt>.
    Sets g.user = {sub, email, name, hd}.
    Optionally enforces GOOGLE_HOSTED_DOMAIN.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = _get_bearer_token()
        if not token:
            return jsonify({"error": "Missing Authorization Bearer token"}), 401

        try:
            # audience check is optional if you want to enforce client_id:
            # set GOOGLE_CLIENT_ID and pass audience=...
            audience = os.getenv("GOOGLE_CLIENT_ID") or None
            req = google_requests.Request()

            if audience:
                claims = id_token.verify_oauth2_token(token, req, audience=audience)
            else:
                # Verify signature + standard claims, but don't enforce aud.
                claims = id_token.verify_oauth2_token(token, req)

            hosted_domain_required = os.getenv("GOOGLE_HOSTED_DOMAIN") or ""
            hd = claims.get("hd") or ""
            if hosted_domain_required and hd != hosted_domain_required:
                return jsonify({"error": "Forbidden: wrong hosted domain"}), 403

            g.user = {
                "sub": claims.get("sub"),
                "email": claims.get("email"),
                "name": claims.get("name"),
                "hd": hd,
            }
            if not g.user["sub"]:
                return jsonify({"error": "Invalid token: missing subject"}), 401

        except Exception:
            # Don't leak validation internals to callers.
            logger.warning("Invalid bearer token", exc_info=True)
            return jsonify({"error": "Invalid token"}), 401

        return fn(*args, **kwargs)

    return wrapper

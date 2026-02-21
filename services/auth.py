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
    token = auth[len("Bearer "):].strip()
    return token or None


def _g_user_is_authenticated() -> bool:
    """
    True if an upstream interceptor already authenticated the request
    (e.g., BFF injected x-user-id and interceptors.py set g.user).
    """
    try:
        u = getattr(g, "user", None)
        return isinstance(u, dict) and bool(u.get("sub"))
    except Exception:
        return False


def require_auth(fn):
    """
    Auth decorator used by routes.

    Behavior:
      1) If g.user is already set (by services/interceptors.py), allow the request.
      2) Otherwise, verify Google ID token from Authorization: Bearer <jwt>.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # If interceptors.py already authenticated, don't re-check.
        if _g_user_is_authenticated():
            return fn(*args, **kwargs)

        token = _get_bearer_token()
        if not token:
            return jsonify({"error": "Missing Authorization Bearer token"}), 401

        try:
            audience = os.getenv("GOOGLE_CLIENT_ID") or None
            req = google_requests.Request()

            if audience:
                claims = id_token.verify_oauth2_token(token, req, audience=audience)
            else:
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
            logger.warning("Invalid bearer token", exc_info=True)
            return jsonify({"error": "Invalid token"}), 401

        return fn(*args, **kwargs)

    return wrapper

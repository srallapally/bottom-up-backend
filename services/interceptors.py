import os
from flask import request, jsonify, g

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from models.session import session_owner_sub


PUBLIC_PATHS = (
    "/api/health",
)


def _public(path: str) -> bool:
    return any(path.startswith(p) for p in PUBLIC_PATHS)


def _token():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return auth[len("Bearer "):].strip()


def install_interceptors(app):

    @app.before_request
    def auth_and_ownership_guard():
        if _public(request.path):
            return None

        token = _token()
        if not token:
            return jsonify({"error": "Missing bearer token"}), 401

        try:
            req = google_requests.Request()
            audience = os.getenv("GOOGLE_CLIENT_ID") or None

            claims = (
                id_token.verify_oauth2_token(token, req, audience=audience)
                if audience else
                id_token.verify_oauth2_token(token, req)
            )

            required_domain = os.getenv("GOOGLE_HOSTED_DOMAIN")
            if required_domain and claims.get("hd") != required_domain:
                return jsonify({"error": "Forbidden domain"}), 403

            g.user = {
                "sub": claims["sub"],
                "email": claims.get("email"),
                "name": claims.get("name"),
            }

        except Exception as e:
            return jsonify({"error": "Invalid token", "details": str(e)}), 401

        # auto-ownership enforcement
        session_id = (request.view_args or {}).get("session_id")
        if session_id:
            owner = session_owner_sub(session_id)
            if not owner:
                return jsonify({"error": "Session missing owner"}), 403
            if owner != g.user["sub"]:
                return jsonify({"error": "Forbidden"}), 403

        return None

import os
import logging
from flask import request, jsonify, g

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from models.session import session_owner_sub

logger = logging.getLogger(__name__)

PUBLIC_PATHS = (
    "/api/health",
)


def _public(path: str) -> bool:
    return any(path.startswith(p) for p in PUBLIC_PATHS)


def _bearer_token() -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth or not auth.startswith("Bearer "):
        return None
    token = auth[len("Bearer "):].strip()
    return token or None


def install_interceptors(app):

    @app.before_request
    def auth_and_ownership_guard():
        # ---- DEBUG (opt-in): log what we received (avoid noisy logs in prod) ----
        auth_debug = (os.getenv("AUTH_DEBUG") or "").lower() in ("1", "true", "yes")
        if auth_debug:
            logger.info(
                "AUTH_DEBUG path=%s host=%s origin=%r auth_present=%s x-user-id=%r x-user-email=%r cookie_present=%s",
                request.path,
                request.headers.get("Host"),
                request.headers.get("Origin"),
                bool(request.headers.get("Authorization")),
                request.headers.get("x-user-id"),
                request.headers.get("x-user-email"),
                bool(request.headers.get("Cookie")),
            )

        if _public(request.path):
            return None

        token = _bearer_token()

        # BFF mode (Express proxy injects x-user-id/x-user-email).
        # SECURITY: Only allow this when a shared secret is configured AND matches.
        bff_user_id = request.headers.get("x-user-id")
        bff_user_email = request.headers.get("x-user-email")
        bff_secret_required = os.getenv("BFF_SHARED_SECRET")
        bff_secret_provided = request.headers.get("x-bff-secret")
        bff_headers_present = bool(bff_user_id or bff_user_email or bff_secret_provided)

        bff_ok = False
        if bff_headers_present:
            if not bff_secret_required:
                # If you want BFF mode, you MUST configure BFF_SHARED_SECRET.
                # FIX 23: log unconditionally — this is a misconfiguration, not a debug detail.
                logger.warning("bff_rejected reason=missing_required_secret path=%s", request.path)
            elif bff_secret_provided != bff_secret_required:
                # FIX 23: log auth failures unconditionally (not gated behind AUTH_DEBUG).
                logger.warning(
                    "bff_rejected reason=secret_mismatch path=%s provided_present=%s",
                    request.path,
                    bff_secret_provided is not None,
                )
            else:
                bff_ok = True

        # ---- AUTH: prefer Bearer token if present; else accept BFF headers (if allowed) ----
        if token:
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
                    # FIX 23: log domain rejection unconditionally.
                    logger.warning(
                        "auth_rejected reason=wrong_domain required=%s got=%s path=%s",
                        required_domain, claims.get("hd"), request.path,
                    )
                    return jsonify({"error": "Forbidden domain", "userMessage": "Forbidden domain"}), 403

                g.user = {
                    "sub": claims["sub"],
                    "email": claims.get("email"),
                    "name": claims.get("name"),
                }

                if auth_debug:
                    logger.info(
                        "AUTH_DEBUG auth_mode=bearer sub=%r email=%r",
                        g.user.get("sub"),
                        g.user.get("email"),
                    )

            except Exception:
                logger.warning("Invalid bearer token", exc_info=True)
                return jsonify({"error": "Invalid token", "userMessage": "Invalid token"}), 401

        else:
            # BFF mode (only if explicitly enabled and secret matches)
            if not bff_ok or not bff_user_id:
                # FIX 23: log unconditionally.
                logger.warning("auth_rejected reason=missing_bearer_token path=%s", request.path)
                return jsonify(
                    {"error": "Missing bearer token", "userMessage": "Missing bearer token"}
                ), 401

            g.user = {
                "sub": bff_user_id,
                "email": bff_user_email,
                "name": None,
            }

            if auth_debug:
                logger.info(
                    "AUTH_DEBUG auth_mode=bff sub=%r email=%r",
                    g.user.get("sub"),
                    g.user.get("email"),
                )

        # ---- Ownership enforcement (only when route contains session_id) ----
        session_id = (request.view_args or {}).get("session_id")
        if session_id:
            # FIX 13: Cache the Firestore owner lookup in g for the request lifetime.
            # Without this, every subrequest or helper that triggers ownership
            # re-evaluation would hit Firestore again for the same session.
            if not hasattr(g, "session_owner_sub_cache"):
                g.session_owner_sub_cache = {}
            if session_id not in g.session_owner_sub_cache:
                g.session_owner_sub_cache[session_id] = session_owner_sub(session_id)
            owner = g.session_owner_sub_cache[session_id]
            if not owner:
                return jsonify({"error": "Session missing owner", "userMessage": "Session missing owner"}), 403
            if owner != g.user["sub"]:
                return jsonify({"error": "Forbidden", "userMessage": "Forbidden"}), 403

        return None
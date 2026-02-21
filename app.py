import logging
import os
from flask import Flask
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from services.interceptors import install_interceptors
from routes.upload import upload_bp
from routes.export import export_bp
from routes.mining import mining_bp
from routes.recommendations import recommendations_bp
from routes.over_provisioned import over_provisioned_bp
from routes.assignments import assignments_bp
from routes.browse import browse_bp

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")

def _parse_cors_origins() -> list:
    # Comma-separated allowlist. Example:
    #   CORS_ORIGINS=http://localhost:5173,http://localhost:3000,https://ui.example.com
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]

    # Safe dev defaults (explicit allowlist; do not use '*').
    return [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
    ]

logging.basicConfig(
    level=(logging.DEBUG if _env_bool("FLASK_DEBUG", False) else logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def create_app():
    app = Flask(__name__)

    # Guardrail: reject oversized request bodies at the framework boundary.
    # Configure via env for GCP deployments.
    # Example: MAX_CONTENT_LENGTH_BYTES=52428800 (50MB)
    try:
        app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH_BYTES", "52428800"))
    except ValueError:
        app.config["MAX_CONTENT_LENGTH"] = 52428800

    # Lock down CORS to an explicit allowlist (env-driven for GCP deployments).
    # NOTE: If the UI only reaches this backend via the Node BFF/proxy, you can
    # tighten this further (or disable CORS entirely) by setting CORS_ORIGINS to the BFF origin only.
    CORS(app, origins=_parse_cors_origins())

    logging.getLogger(__name__).info(
        "BOOT env BFF_SHARED_SECRET present=%s",
        bool(os.getenv("BFF_SHARED_SECRET"))
    )
    install_interceptors(app)

    # V1 blueprints (existing)
    app.register_blueprint(upload_bp)
    app.register_blueprint(mining_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(recommendations_bp)
    app.register_blueprint(over_provisioned_bp)

    app.register_blueprint(assignments_bp)
    app.register_blueprint(browse_bp)

    if _env_bool("FLASK_DEBUG", False):
        print(f"Registered routes: {[rule.rule for rule in app.url_map.iter_rules()]}")

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_entity_too_large(e):
        return {
            "error": "Request too large",
            "max_bytes": app.config.get("MAX_CONTENT_LENGTH"),
        }, 413

    @app.route("/api/health", methods=["GET"])
    def health():
        return {
            "status": "ok",
            "v1": "available",
            "v2": "available",
        }

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(
        debug=_env_bool("FLASK_DEBUG", True),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
    )

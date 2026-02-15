from flask import Flask
from flask_cors import CORS

from routes.upload import upload_bp
from routes.export import export_bp
from routes.mining import mining_bp
from routes.recommendations import recommendations_bp
from routes.over_provisioned import over_provisioned_bp

# V2 routes (hybrid approach)
try:

    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False


def create_app():
    app = Flask(__name__)
    CORS(app)

    # V1 blueprints (existing)
    app.register_blueprint(upload_bp)
    app.register_blueprint(mining_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(recommendations_bp)
    app.register_blueprint(over_provisioned_bp)

    @app.route("/api/health", methods=["GET"])
    def health():
        return {
            "status": "ok",
            "v1": "available",
            "v2": "available" if V2_AVAILABLE else "unavailable (install python-igraph leidenalg)",
        }

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
import logging
from flask import Flask
from flask_cors import CORS
from services.interceptors import install_interceptors
from routes.upload import upload_bp
from routes.export import export_bp
from routes.mining import mining_bp
from routes.recommendations import recommendations_bp
from routes.over_provisioned import over_provisioned_bp
from routes.assignments import assignments_bp
from routes.browse import browse_bp

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_app():
    app = Flask(__name__)
    CORS(app)
    install_interceptors(app)
    # V1 blueprints (existing)
    app.register_blueprint(upload_bp)
    app.register_blueprint(mining_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(recommendations_bp)
    app.register_blueprint(over_provisioned_bp)

    app.register_blueprint(assignments_bp)

    app.register_blueprint(browse_bp)
    print(f"Registered routes: {[rule.rule for rule in app.url_map.iter_rules()]}")
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
    app.run(debug=True, port=5000,host='0.0.0.0')
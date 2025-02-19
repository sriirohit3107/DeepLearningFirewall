from flask import request, jsonify
from app_py.model import is_malicious_request

def init_routes(app):
    @app.route('/')
    def home():
        return "AI Firewall Home Page"

    @app.route('/detect', methods=['POST'])
    def detect():
        data = request.json
        http_request = data.get('http_request')
        malicious = is_malicious_request(http_request)
        return jsonify({'malicious': malicious})
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import base64
import json
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_available_models():
    """動態取得目前 API Key 可用的所有現役模型，優先嘗試輕量/備援模型以避開已滿額度"""
    preferred = [
        "gemini-2.5-flash-lite",
        "gemini-3.8-flash-lite",
        "gemini-2.5-pro",
        "gemini-3.8-pro",
        "gemini-3.8-flash",
    ]
    available = []
    try:
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                name = m.name.replace("models/", "")
                available.append(name)
    except Exception:
        pass

    ordered = [m for m in preferred if m in available]
    for m in available:
        if m not in ordered and "embedding" not in m and "imagen" not in m:
            ordered.append(m)
    return ordered or preferred


@app.route("/", methods=["GET"])
def serve_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return send_file(index_path)
    return "找不到 index.html", 404


@app.route("/api/gemini", methods=["POST", "OPTIONS"])
@app.route("/api/<path:subpath>", methods=["POST", "OPTIONS"])
def gemini_proxy(subpath=None):
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if not GEMINI_API_KEY:
        return jsonify({
            "success": False,
            "error": "伺服器尚未設定 GEMINI_API_KEY 環境變數，請到 Vercel 設定後重新部署。"
        }), 500

    try:
        data = request.get_json(silent=True) or {}

        prompt = data.get("prompt") or data.get("text") or data.get("query") or ""
        images = data.get("images") or []
        json_mode = bool(data.get("json_mode") or data.get("jsonMode"))

        raw_contents = data.get("contents")
        contents = []

        if raw_contents and isinstance(raw_contents, list):
            contents = raw_contents
        else:
            if prompt:
                contents.append(prompt)

            for img in images:
                if isinstance(img, dict) and "data" in img and "mime_type" in img:
                    contents.append({
                        "mime_type": img["mime_type"],
                        "data": base64.b64decode(img["data"])
                    })
                elif isinstance(img, str) and "," in img:
                    header, encoded = img.split(",", 1)
                    mime_type = header.split(";")[0].replace("data:", "") or "image/jpeg"
                    contents.append({
                        "mime_type": mime_type,
                        "data": base64.b64decode(encoded)
                    })

        if not contents:
            return jsonify({
                "success": False,
                "error": f"未收到有效的請求內容（收到的欄位：{list(data.keys())}）"
            }), 400

        generation_config = {}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        response = None
        last_error = None

        for model_name in get_available_models():
            try:
                model = genai.GenerativeModel(
                    model_name,
                    generation_config=generation_config
                )
                response = model.generate_content(contents)
                break
            except Exception as model_err:
                last_error = model_err
                err_msg = str(model_err)
                if "429" in err_msg or "404" in err_msg or "quota" in err_msg.lower() or "not found" in err_msg.lower():
                    continue
                raise model_err

        if response is None:
            raise last_error or RuntimeError("目前所有可用模型皆已達今日上限")

        result_text = response.text or ""

        payload = {
            "success": True,
            "text": result_text,
            "result": result_text,
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": result_text}]
                    }
                }
            ]
        }

        if json_mode:
            try:
                payload["data"] = json.loads(result_text)
            except json.JSONDecodeError:
                pass

        return jsonify(payload)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

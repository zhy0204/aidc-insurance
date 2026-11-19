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

# 免費額度按「個別模型」獨立計算，主模型達 20 次上限時自動依序切換備援模型
FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.0-flash",
    "gemini-2.0-flash",
]


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

        for model_name in FALLBACK_MODELS:
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
                # 遇到 429 額度滿或 404 模型不支援時，自動跳下一個備援模型
                if "429" in err_msg or "404" in err_msg or "quota" in err_msg.lower():
                    continue
                raise model_err

        if response is None:
            raise last_error or RuntimeError("所有備援模型目前皆無法使用")

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

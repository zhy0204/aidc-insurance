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


@app.route("/", methods=["GET"])
def serve_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return send_file(index_path)
    return "找不到 index.html", 404


# 同時接住所有可能的 /api/... 路徑，避免前端網址對不上而回傳 404 HTML
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

        # 自動相容不同前端欄位名稱：prompt / text / query / contents
        prompt = data.get("prompt") or data.get("text") or data.get("query") or ""
        images = data.get("images") or []
        json_mode = bool(data.get("json_mode") or data.get("jsonMode"))

        # 若前端直接傳送 Gemini 原生格式的 contents
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

        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config=generation_config
        )

        response = model.generate_content(contents)
        result_text = response.text or ""

        # 同時回傳 text 與 Gemini 原生 candidates 結構，確保前端無論讀哪個欄位都拿得到結果
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

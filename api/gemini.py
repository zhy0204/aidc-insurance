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

# 取得專案根目錄的絕對路徑，確保在 Vercel 環境能穩定讀取 index.html
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.route("/", methods=["GET"])
def serve_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return send_file(index_path)
    return "找不到 index.html，請確認它位於儲存庫根目錄。", 404


@app.route("/api/gemini", methods=["POST"])
def gemini_proxy():
    if not GEMINI_API_KEY:
        return jsonify({
            "success": False,
            "error": "伺服器尚未設定 GEMINI_API_KEY 環境變數"
        }), 500

    try:
        data = request.get_json() or {}
        prompt = data.get("prompt", "")
        images = data.get("images", [])
        json_mode = data.get("json_mode", False)

        generation_config = {}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config=generation_config
        )

        contents = []
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
                "error": "未收到有效的 prompt 或圖片內容"
            }), 400

        response = model.generate_content(contents)
        result_text = response.text or ""

        if json_mode:
            try:
                parsed = json.loads(result_text)
                return jsonify({"success": True, "data": parsed, "text": result_text})
            except json.JSONDecodeError:
                return jsonify({"success": True, "text": result_text})

        return jsonify({"success": True, "text": result_text})

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

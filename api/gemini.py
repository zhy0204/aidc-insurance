from flask import Flask, jsonify, request
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

@app.route('/api/gemini', methods=['POST'])
def gemini_proxy():
    if not GEMINI_API_KEY:
        return jsonify({"success": False, "error": "伺服器尚未設定 GEMINI_API_KEY 環境變數"}), 500

    try:
        data = request.get_json() or {}
        prompt = data.get('prompt', '')
        images = data.get('images', [])
        json_mode = data.get('json_mode', False)

        generation_config = {}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        model = genai.GenerativeModel('gemini-2.5-flash')
        
        content_parts = [prompt]
        for img_base64 in images:
            if ',' in img_base64:
                img_base64 = img_base64.split(',')[1]
            image_bytes = base64.b64decode(img_base64)
            content_parts.append({
                "mime_type": "image/jpeg",
                "data": image_bytes
            })

        response = model.generate_content(content_parts, generation_config=generation_config)
        res_text = response.text

        if json_mode:
            cleaned = res_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            res_data = json.loads(cleaned.strip())
            return jsonify({"success": True, "data": res_data})
        else:
            return jsonify({"success": True, "data": res_text})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

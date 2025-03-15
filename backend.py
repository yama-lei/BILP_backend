from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# 假设你已经配置好 BLIP 模型
from transformers import BlipProcessor, BlipForQuestionAnswering

app = Flask(__name__)
CORS(app)  # 启用 CORS

model_path = "blip-vqa-base"
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForQuestionAnswering.from_pretrained(model_path)

@app.route("/vqa", methods=["POST"])
def vqa():
    image_file = request.files.get("image")
    question = request.form.get("question")
    if image_file is None or not question:
        return jsonify({"error": "缺少图片或问题"}), 400

    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs,max_length=100)
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

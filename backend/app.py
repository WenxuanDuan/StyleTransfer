from flask import Flask, request, jsonify
from flask_cors import CORS
from StyleTransfer.backend.style_transfer import run_style_transfer
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './model/images/content/'
RESULT_FOLDER = './model/images/results/'
STYLE_IMAGES = './model/images/style/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/styles', methods=['GET'])
def get_styles():
    # 返回可用的风格选项
    styles = [f for f in os.listdir(STYLE_IMAGES) if f.endswith('.jpg')]
    return jsonify(styles)

@app.route('/upload', methods=['POST'])
def upload_image():
    # 接收用户上传的图像和风格选项
    content_image = request.files['content']
    style_name = request.form['style']

    content_path = os.path.join(UPLOAD_FOLDER, content_image.filename)
    result_path = os.path.join(RESULT_FOLDER, f'result_{content_image.filename}')

    content_image.save(content_path)

    style_image_path = os.path.join(STYLE_IMAGES, style_name)
    result_image = run_style_transfer(content_path, style_image_path)

    # 保存生成的图像
    result_image.save(result_path)

    return jsonify({'result_url': f'/results/{os.path.basename(result_path)}'})

@app.route('/results/<filename>', methods=['GET'])
def get_result(filename):
    return app.send_static_file(os.path.join(RESULT_FOLDER, filename))

if __name__ == "__main__":
    app.run(debug=True)

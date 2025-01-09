import base64
import tempfile

from flask import Flask, render_template, request
from vertexai.preview.vision_models import ImageGenerationModel


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":    # No form submitted
        return render_template("index.html")

    # Should be POST with a form submitted
    prompt = request.form["prompt"]
    image_data = generate_image(prompt)
    image_url = get_url(image_data)
    return render_template("index.html", prompt=prompt, image_url=image_url)


def generate_image(prompt):
    model = ImageGenerationModel.from_pretrained("imagegeneration@002")
    response = model.generate_images(prompt=prompt)[0]
    with tempfile.NamedTemporaryFile("wb") as f:
        filename = f.name
        response.save(filename, include_generation_parameters=False)
        with open(filename, "rb") as image_file:
            binary_image = image_file.read()
    return binary_image


def get_url(image_data):
    base64_image = base64.b64encode(image_data).decode("utf-8")
    content_type = "image/png"
    image_uri = f"data:{content_type};base64,{base64_image}"
    return image_uri

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

from flask import Flask, render_template, request
import numpy as np
import cv2
from image_scanner import scan_image_to_text

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file was uploaded
        if 'document' not in request.files:
            return render_template("result.html", text="⚠️ No file part")
        
        file = request.files['document']
        if file.filename == "":
            return render_template("result.html", text="⚠️ No file selected")
        
        # Read image from upload
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return render_template("result.html", text="❌ Invalid image format")
        
        # Scan and extract text
        extracted_text = scan_image_to_text(img)
        return render_template("result.html", text=extracted_text)

    # Show upload form on GET
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

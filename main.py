from flask import Flask, request, jsonify
from app.services.process_image import ProcessImage


app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Welcome to the License Plate Detection API"})

@app.route("/detect-plate", methods=["POST"])
def detect_plate_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    proccess_image = ProcessImage(file)
    results = proccess_image.exec()
    return jsonify({"data": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

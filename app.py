from flask import Flask, render_template, request, send_file
from graphs import generate_graphs
from predict import predict_soil

import os

# PDF generator
from report import create_pdf_report

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Store latest result globally
latest_result = None
latest_image_path = None

# -----------------------------
# HOME ROUTE
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    global latest_result
    global latest_image_path

    uploaded_image = None
    result = None

    if request.method == "POST":

        uploaded_file = request.files["soil_image"]

        if uploaded_file.filename != "":

            # Save uploaded image
            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                uploaded_file.filename
            )

            uploaded_file.save(file_path)

            uploaded_image = uploaded_file.filename

            # Predict soil
            result = predict_soil(file_path)

            # Generate graphs
            generate_graphs(result)

            # Store latest data
            latest_result = result
            latest_image_path = file_path

    return render_template(
        "index.html",
        uploaded_image=uploaded_image,
        result=result
    )

# -----------------------------
# PDF DOWNLOAD ROUTE
# -----------------------------

@app.route("/download_report")
def download_report():

    global latest_result
    global latest_image_path

    # Generate PDF
    pdf_path = create_pdf_report(
        latest_result,
        latest_image_path
    )

    # Send PDF file
    return send_file(
        pdf_path,
        as_attachment=True
    )

# -----------------------------
# RUN APP
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)
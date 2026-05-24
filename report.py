from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

from reportlab.platypus.tables import Table
from reportlab.platypus.tables import TableStyle

from reportlab.lib import colors

import os

def create_pdf_report(result, image_path):

    # PDF file path
    pdf_path = "soil_analysis_report.pdf"

    # Create document
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter
    )

    # Styles
    styles = getSampleStyleSheet()

    # Elements list
    elements = []

    # -----------------------------
    # TITLE
    # -----------------------------

    title = Paragraph(
        "Soil Analysis Report",
        styles['Title']
    )

    elements.append(title)
    elements.append(Spacer(1, 20))

    # -----------------------------
    # SOIL IMAGE
    # -----------------------------

    if os.path.exists(image_path):

        soil_image = Image(
            image_path,
            width=250,
            height=180
        )

        elements.append(soil_image)
        elements.append(Spacer(1, 20))

    # -----------------------------
    # ANALYSIS TABLE
    # -----------------------------

    table_data = [

        ["Parameter", "Result"],

        ["Soil Type", result["soil_type"]],

        ["Prediction Confidence", f'{result["confidence"]}%'],

        ["Fertility Level", result["fertility"]],

        ["Moisture Level", result["moisture"]],

        ["Soil Health Score", f'{result["health_score"]}%'],

        ["Recommended Crops", ", ".join(result["crops"])],

        ["Suggestion", result["suggestion"]]
    ]

    table = Table(
        table_data,
        colWidths=[200, 300]
    )

    # Table styling
    table.setStyle(TableStyle([

        ('BACKGROUND', (0,0), (-1,0), colors.green),

        ('TEXTCOLOR', (0,0), (-1,0), colors.white),

        ('GRID', (0,0), (-1,-1), 1, colors.black),

        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),

        ('BOTTOMPADDING', (0,0), (-1,0), 10),

        ('BACKGROUND', (0,1), (-1,-1), colors.beige)

    ]))

    elements.append(table)

    elements.append(Spacer(1, 30))

    # -----------------------------
    # BAR GRAPH
    # -----------------------------

    bar_graph_path = "static/graphs/bar_graph.png"

    if os.path.exists(bar_graph_path):

        bar_graph = Image(
            bar_graph_path,
            width=400,
            height=250
        )

        elements.append(bar_graph)

        elements.append(Spacer(1, 20))

    # -----------------------------
    # PIE CHART
    # -----------------------------

    pie_chart_path = "static/graphs/pie_chart.png"

    if os.path.exists(pie_chart_path):

        pie_chart = Image(
            pie_chart_path,
            width=300,
            height=300
        )

        elements.append(pie_chart)

    # -----------------------------
    # BUILD PDF
    # -----------------------------

    doc.build(elements)

    return pdf_path
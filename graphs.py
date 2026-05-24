import matplotlib

# Use non-GUI backend
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def generate_graphs(result):

    # Fertility values
    fertility_map = {
        "Low": 40,
        "Medium": 70,
        "High": 90
    }

    moisture_map = {
        "Low": 30,
        "Moderate": 60,
        "High": 90
    }

    fertility_value = fertility_map[result["fertility"]]
    moisture_value = moisture_map[result["moisture"]]

    # -----------------------------
    # BAR GRAPH
    # -----------------------------

    labels = ["Health Score", "Fertility", "Moisture"]

    values = [
        result["health_score"],
        fertility_value,
        moisture_value
    ]

    plt.figure(figsize=(6, 4))

    plt.bar(labels, values)

    plt.ylim(0, 100)

    plt.title("Soil Analysis Graph")

    plt.ylabel("Percentage")

    # Save graph
    plt.savefig(
        "static/graphs/bar_graph.png",
        bbox_inches='tight'
    )

    plt.close()

    # -----------------------------
    # PIE CHART
    # -----------------------------

    pie_values = [
        fertility_value,
        moisture_value,
        100 - fertility_value
    ]

    pie_labels = [
        "Fertility",
        "Moisture",
        "Other Factors"
    ]

    plt.figure(figsize=(5, 5))

    plt.pie(
        pie_values,
        labels=pie_labels,
        autopct="%1.1f%%"
    )

    plt.title("Soil Composition")

    # Save pie chart
    plt.savefig(
        "static/graphs/pie_chart.png",
        bbox_inches='tight'
    )

    plt.close()
import dash
from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from dash.dependencies import ALL
from dash import callback_context

"""
This script creates a Dash app that allows users to browse a list of patent summaries and classify them into industry categories.
"""

df = pd.read_csv('sum_2024_sample.tsv', sep='\t')

"""
Here I used ChatGPT to generate a list of dictionaries containing industry titles and descriptions. 
The idea is to embed these descriptions and use them as reference points for classifying patent summaries
through cosine similarity.
"""

retrieval_corpus = [
    {"title": "Technology", "text": "Innovations related to software development, artificial intelligence, machine learning, data processing, and digital systems."},
    {"title": "Healthcare", "text": "Advancements in medical devices, diagnostics, pharmaceuticals, biotechnology, and healthcare delivery systems."},
    {"title": "Automotive", "text": "Technologies involving vehicles, autonomous driving systems, electric vehicles, and transportation infrastructure."},
    {"title": "Energy", "text": "Patents covering renewable energy, solar power, wind turbines, oil and gas exploration, and energy efficiency solutions."},
    {"title": "Manufacturing", "text": "Industrial processes, robotics, automation, supply chain systems, and machinery used in production."},
    {"title": "Consumer Electronics", "text": "Devices such as smartphones, televisions, wearable technology, and home automation systems."},
    {"title": "Agriculture", "text": "Innovations in farming equipment, crop management, irrigation systems, and sustainable agriculture practices."},
    {"title": "Telecommunications", "text": "Wireless communication technologies, network infrastructure, 5G, and internet-of-things (IoT) devices."},
    {"title": "Aerospace", "text": "Technologies for aircraft, spacecraft, satellite systems, and aviation safety equipment."},
    {"title": "Construction", "text": "Building materials, structural engineering, smart buildings, and construction equipment."},
    {"title": "Food and Beverage", "text": "Production, processing, and packaging of food items, as well as innovations in food safety and nutrition."},
    {"title": "Biotechnology", "text": "Applications of biological systems and organisms in pharmaceuticals, genetic engineering, and industrial processes."},
    {"title": "Chemicals", "text": "Production of industrial chemicals, polymers, specialty chemicals, and innovations in material science."},
    {"title": "Education", "text": "Learning platforms, e-learning tools, educational technologies, and methods to enhance learning outcomes."},
    {"title": "Finance", "text": "Financial services technologies, blockchain, payment systems, and fraud detection."},
    {"title": "Retail", "text": "E-commerce systems, inventory management, customer experience solutions, and logistics innovations."},
    {"title": "Logistics and Transportation", "text": "Supply chain management, freight solutions, and innovations in transportation networks."},
    {"title": "Environmental Science", "text": "Solutions for waste management, water purification, air quality monitoring, and ecosystem conservation."},
    {"title": "Defense and Security", "text": "Technologies for military equipment, cybersecurity, surveillance, and personal safety."},
    {"title": "Entertainment and Media", "text": "Streaming platforms, content production, virtual reality, gaming, and multimedia systems."},
    {"title": "Textiles and Apparel", "text": "Innovations in fabric production, wearable materials, and sustainable fashion."},
    {"title": "Mining and Materials", "text": "Extraction technologies, metallurgy, and the development of advanced materials."},
    {"title": "Real Estate", "text": "Smart city planning, real estate technology platforms, and property management solutions."},
    {"title": "Pharmaceuticals", "text": "Drug discovery, production processes, and delivery mechanisms for therapeutic agents."},
    {"title": "Insurance", "text": "Risk management systems, predictive analytics, and insurance technology solutions."},
    {"title": "Maritime", "text": "Shipbuilding, navigation systems, and technologies related to ocean exploration and shipping logistics."},
    {"title": "Sports and Recreation", "text": "Sports equipment, fitness technologies, and innovations in leisure activities."}
]

# Load the sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Generate embeddings for the industry descriptions
industry_texts = [entry["text"] for entry in retrieval_corpus]
industry_titles = [entry["title"] for entry in retrieval_corpus]
industry_embeddings = model.encode(industry_texts)

# Function to classify the industry of a patent summary
def classify_industry(summary_text):
    summary_embedding = model.encode([summary_text])
    similarities = cosine_similarity(summary_embedding, industry_embeddings)
    sorted_indices = similarities.argsort()[0][::-1]
    top_industries = [industry_titles[i] for i in sorted_indices[:5]]
    top_similarities = similarities[0][sorted_indices[:5]]
    return top_industries, top_similarities

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Patent Industry Classifier"

# Layout
def generate_patent_list():
    return [
        dbc.ListGroupItem(
            str(patent_id),
            id={"type": "patent-item", "index": patent_id},
            action=True,
        )
        for patent_id in df['patent_id']
    ]

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Patent Browser"),
            html.Br(),
            dbc.ListGroup(
                id="patent-list", children=generate_patent_list(), style={"height": "200px", "overflowY": "scroll"}
            ),
            html.Br(),
            html.Div(
                id="summary-text-display-container",  # Container for the summary text
                children=[
                    dcc.Markdown(
                        id="summary-text-display",
                        style={
                            "width": "100%",          # Full width
                            "height": "150px",        # Fixed height
                            "overflowY": "scroll",    # Vertical scrolling for overflow
                            "border": "1px solid #ccc",  # Optional border for visibility
                            "padding": "10px",        # Padding inside the box
                            "backgroundColor": "#f9f9f9",  # Optional background color
                        },
                    )
                ],
                style={"marginTop": "20px"}  
            )
        ], width=3, style={"borderRight": "1px solid #ddd", "paddingRight": "20px"}),

        dbc.Col([
            html.H3("Classification Results"),
            dcc.Graph(id="output-plot"),
        ], width=7),
    ], justify="center"),
], fluid=True)

# Callback functions 
@app.callback(
    [Output("output-plot", "figure"),
     Output("summary-text-display", "children")],
    Input({"type": "patent-item", "index": ALL}, "n_clicks"),
    State({"type": "patent-item", "index": ALL}, "id"),
    prevent_initial_call=True
)
def update_output(n_clicks, patent_ids):
    # Use dash.callback_context to track the triggered input
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    # Get the index of the clicked patent item
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    clicked_patent_index = eval(triggered_id)["index"]

    # Retrieve summary text and classify
    summary_text = df[df['patent_id'] == clicked_patent_index]['summary_text'].values[0]
    industries, probabilities = classify_industry(summary_text)

    # Create a DataFrame for visualization
    result_df = pd.DataFrame({"Industry": industries, "Probability": probabilities})

    # Generate a pie chart
    fig = px.pie(result_df, names="Industry", values="Probability",
                 title="Top Matching Industries", color="Industry",
                 color_discrete_sequence=px.colors.qualitative.Set3)

    fig.update_layout(
        legend=dict(
            title="Industry",
            itemwidth=30,
            traceorder='normal',
            font=dict(size=10),
            itemsizing='constant',
            x=1.05,
            y=1,
            orientation='v',
        ),
        autosize=False,
        width=400,
        height=400,
        margin=dict(l=0, r=200, t=50, b=50),
    )

    # Return the figure and the summary text
    return fig, f"**Summary Text**: {summary_text}"

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

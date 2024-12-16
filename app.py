import numpy as np
import dash
from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from dash.dependencies import ALL
from dash import callback_context
from keybert import KeyBERT
from dash.exceptions import PreventUpdate
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re

# Loading 0.01% sample of the patent dataset.
df = pd.read_csv('sum_2024_sample.tsv', sep='\t')

"""
retreival_corpus was created by ChatGPT as a response to the following prompt:

"Write me retrieval corpus for a patent industry classifier. The corpus should contain a list of dictionaries, 
where each dictionary has two keys: 'title' and 'text'. The 'title' key should contain the name of an industry, 
and the 'text' key should contain a description of that industry. 
The corpus should cover a diverse range of industries."

"""

retrieval_corpus = [
    {
    "title": "Data Science",
    "text": "Data science and machine learning involve extracting insights from large datasets, developing predictive models, and creating algorithms for automation. Key areas include natural language processing, computer vision, reinforcement learning, deep learning, data visualization, predictive analytics, algorithm development, and AI applications across various industries such as finance, healthcare, marketing, and transportation."
    },
    {
        "title": "Technology", 
        "text": "Breakthroughs in software engineering, cloud computing, artificial intelligence, machine learning algorithms, blockchain, quantum computing, digital communication systems, and virtual/augmented reality applications. Focuses on the design, development, and optimization of digital infrastructures and computing platforms."
    },
    {
        "title": "Healthcare", 
        "text": "Advancements in medical imaging, precision medicine, surgical robotics, telemedicine, wearable health devices, pharmaceutical innovations, genetic therapies, diagnostic tools, and healthcare management systems aimed at improving patient outcomes and reducing costs."
    },
    {
        "title": "Automotive", 
        "text": "Developments in electric and hybrid vehicles, autonomous navigation systems, advanced vehicle safety technologies, engine optimization, sustainable materials for car manufacturing, and innovations in intelligent transportation systems and infrastructure connectivity."
    },
    {
        "title": "Energy", 
        "text": "Progress in renewable energy sources such as solar panels, wind turbines, geothermal systems, battery storage technologies, nuclear energy advancements, smart grids, energy conservation methodologies, and exploration of alternative fuels like hydrogen."
    },
    {
        "title": "Manufacturing", 
        "text": "Innovations in additive manufacturing (3D printing), smart factories, industrial robotics, predictive maintenance, Internet of Things (IoT)-enabled production lines, advanced supply chain logistics, and automated quality control systems for high-precision manufacturing."
    },
    {
        "title": "Consumer Electronics", 
        "text": "Technologies for smart home automation, personal electronics like smartphones and tablets, virtual assistants, wearables, entertainment systems, gaming consoles, IoT-connected devices, and advancements in miniaturization and battery efficiency."
    },
    {
        "title": "Agriculture", 
        "text": "Revolutionary farming practices including drone-assisted crop monitoring, precision irrigation systems, soil analysis technologies, genetically engineered crops, sustainable pest control, vertical farming, and innovations in farm machinery and robotics."
    },
    {
        "title": "Telecommunications", 
        "text": "Innovations in wireless communication, 5G network infrastructure, satellite internet systems, IoT networks, advanced signal processing, fiber-optic technologies, secure communication protocols, and enhanced mobile device connectivity solutions."
    },
    {
        "title": "Aerospace", 
        "text": "Breakthroughs in space exploration technologies, satellite communication systems, aviation safety, unmanned aerial vehicles (drones), propulsion systems, composite materials for aircraft, and development of interplanetary travel solutions."
    },
    {
        "title": "Construction", 
        "text": "Advancements in sustainable building materials, modular construction, smart building technologies, green architecture, 3D-printed structures, construction robotics, structural engineering analysis, and innovations in urban infrastructure planning."
    },
    {
        "title": "Food and Beverage", 
        "text": "Research in food processing technologies, novel food preservation methods, sustainable packaging, nutritional optimization, alternative protein sources like plant-based and lab-grown meat, food safety systems, and beverage formulation innovations."
    },
    {
        "title": "Biotechnology", 
        "text": "Applications of genomics, bioinformatics, genetic engineering, synthetic biology, biopharmaceutical development, regenerative medicine, bioprocess engineering, and the creation of environmentally sustainable bio-based products."
    },
    {
        "title": "Chemicals", 
        "text": "Developments in petrochemicals, green chemistry practices, advanced polymer materials, industrial catalysts, nanomaterials, specialty coatings, chemical recycling technologies, and new methods for reducing the environmental impact of chemical production."
    },
    {
        "title": "Education", 
        "text": "Technologies for online learning platforms, virtual classrooms, interactive educational tools, gamification in learning, artificial intelligence-driven tutoring systems, personalized learning analytics, and advancements in pedagogical methodologies."
    },
    {
        "title": "Finance", 
        "text": "Innovations in digital payment systems, blockchain-based financial solutions, algorithmic trading, risk assessment tools, financial fraud detection, decentralized finance (DeFi) platforms, and advancements in personal finance management apps."
    },
    {
        "title": "Retail", 
        "text": "E-commerce optimization technologies, virtual shopping experiences, inventory management software, last-mile delivery solutions, customer behavior analytics, omnichannel retail strategies, and supply chain efficiency enhancements."
    },
    {
        "title": "Logistics and Transportation", 
        "text": "Advancements in freight management systems, route optimization algorithms, autonomous delivery systems, drone logistics, cold chain technologies, smart warehousing, and sustainable transportation solutions."
    },
    {
        "title": "Environmental Science", 
        "text": "Innovative approaches to renewable resource management, water desalination, air purification systems, carbon capture technologies, waste-to-energy processes, biodiversity conservation methods, and sustainable urban planning initiatives."
    },
    {
        "title": "Defense and Security", 
        "text": "Breakthroughs in cybersecurity frameworks, advanced surveillance technologies, autonomous defense systems, threat detection AI, counter-drone systems, biometric authentication, and innovations in personal and national security equipment."
    },
    {
        "title": "Entertainment and Media", 
        "text": "Innovations in digital content streaming, virtual production tools, immersive virtual reality (VR) and augmented reality (AR) experiences, gaming engines, interactive storytelling, and new distribution platforms for creative content."
    },
    {
        "title": "Textiles and Apparel", 
        "text": "Advances in smart fabrics, wearable technology, sustainable textile manufacturing, recycled materials, 3D knitting technology, fast fashion logistics, and design innovations for performance-oriented clothing."
    },
    {
        "title": "Mining and Materials", 
        "text": "Progress in mineral exploration technologies, sustainable mining practices, advanced metallurgy, composite material development, rare earth element processing, and innovations in resource extraction and refinement."
    },
    {
        "title": "Real Estate", 
        "text": "Smart property management platforms, innovations in real estate investment analysis, green building certifications, virtual property tours, urban planning technologies, and advancements in sustainable housing solutions."
    },
    {
        "title": "Pharmaceuticals", 
        "text": "Research in drug discovery pipelines, biologics manufacturing, personalized medicine, controlled release drug delivery systems, vaccine development, and innovative methods for pharmaceutical formulation and production."
    },
    {
        "title": "Insurance", 
        "text": "Predictive analytics for risk assessment, advancements in insurtech platforms, automated claims processing, customer experience innovations, usage-based insurance models, and AI-powered fraud detection tools."
    },
    {
        "title": "Maritime", 
        "text": "Technological developments in marine navigation, shipbuilding innovations, sustainable shipping practices, ocean exploration equipment, offshore wind energy systems, and advancements in underwater robotics."
    },
    {
        "title": "Sports and Recreation", 
        "text": "Innovations in wearable fitness trackers, smart sports equipment, virtual coaching systems, stadium technologies, e-sports platforms, and advancements in materials for athletic performance optimization."
    }
]

"""
Loading different models:
- SentenceTransformer: A transformer-based model for encoding sentences into embeddings of size 384.
- KeyBERT: A model for extracting keywords from text using BERT embeddings.
- WordNetLemmatizer: A lemmatizer from the NLTK library.
"""

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
kw_model = KeyBERT()
lemmatizer = WordNetLemmatizer()

# Downloading necessary NLTK resources for text preprocessing
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess text by removing stopwords and lemmatizing words
def remove_stopwords_and_lemmatize(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    # Tokenize the text
    words = word_tokenize(text)
    # Lemmatize each word and remove stopwords
    lemmatized_words = [
        lemmatizer.lemmatize(word) 
        for word in words if word.lower() not in stop_words
    ]
    # Return the lemmatized, stopword-free text
    return ' '.join(lemmatized_words)

# Using stopwords/lemmatization function and SentenceTransformer model for embedding the retrieval corpus text.
industry_texts = [remove_stopwords_and_lemmatize(entry["text"]) for entry in retrieval_corpus]
industry_titles = [entry["title"] for entry in retrieval_corpus]
industry_embeddings = model.encode(industry_texts)

"""
Function to classify the summary text of a patent into the top industries based on cosine similarity between embeddings of
the summary and the industry descriptions. The function returns the top industries and their similarity scores.
If no industry has a similarity score above the threshold, the function returns the top clusters based on similarity.
"""

def classify_industry(summary_text, num_top_clusters=5, similarity_threshold=0.4):
    clean_text = remove_stopwords_and_lemmatize(summary_text)
    summary_embedding = model.encode([clean_text])
    similarities = cosine_similarity(summary_embedding, industry_embeddings)
    sorted_indices = similarities.argsort()[0][::-1]

    top_indices, top_similarities = [], []
    count = 0
    for idx in sorted_indices:
        if similarities[0][idx] > similarity_threshold and count < num_top_clusters:
            top_indices.append(idx)
            top_similarities.append(similarities[0][idx])
            count += 1
    if len(top_indices) < num_top_clusters:
        for idx in sorted_indices[len(top_indices):num_top_clusters]:
            top_indices.append(idx)
            top_similarities.append(similarities[0][idx])

    top_industries = [industry_titles[i] for i in top_indices]
    return top_industries, top_similarities


"""
Initializing the Dash app and creating the layout with the following components:
- A list of patents for browsing.
- A summary text display.
- A keyword list display.
- A classification results plot.
"""

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Patent Industry Classifier"

# Function to generate the list of patents for browsing
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
    # Row for patent list and classification results
    dbc.Row([
        dbc.Col([
            html.H3("Patent Browser"),
            html.Br(),
            dbc.ListGroup(
                id="patent-list", children=generate_patent_list(),
                style={"height": "200px", "overflowY": "scroll"}
            ),
        ], width=4, style={"borderRight": "1px solid #ddd", "paddingRight": "20px"}),

        dbc.Col([
            html.H3("Classification Results"),
            dcc.Graph(id="output-plot"),
        ], width=8),
    ]),

    # Row for patent summary and keywords
    dbc.Row([
        dbc.Col([
            html.H3("Patent Summary"),
            html.Div(
                id="summary-text-display-container",
                children=[
                    dcc.Markdown(
                        id="summary-text-display",
                        style={
                            "width": "100%", "height": "150px", "overflowY": "scroll",
                            "border": "1px solid #ccc", "padding": "10px", "backgroundColor": "#f9f9f9",
                        },
                    )
                ],
                style={"marginTop": "20px"}
            ),
        ], width=4, style={"borderRight": "1px solid #ddd", "paddingRight": "20px"}),

        dbc.Col([
            html.H3("Keywords"),
            html.Div(
                id="keyword-list-container",
                children=[
                    html.Ul(id="keyword-list", style={"marginLeft": "10px"})
                ],
                style={"marginTop": "20px"}
            )
        ], width=6), 
    ])

], fluid=True)

"""
Callback to update the patent summary, classification results, and keyword list when a patent is clicked.
The callback retrieves the selected patent data, classifies the summary text into top industries, and extracts keywords.
"""


@app.callback(
    [
        Output("summary-text-display", "children"),
        Output("output-plot", "figure"),
        Output("keyword-list", "children")
    ],
    [Input({"type": "patent-item", "index": ALL}, "n_clicks")]
)

def update_patent_summary_and_plot(n_clicks):
    ctx = callback_context
    if not ctx.triggered or all(click is None for click in n_clicks):
        raise PreventUpdate

    # Get the clicked patent_id
    clicked_id = ctx.triggered[0]["prop_id"].split(".")[0]
    clicked_id = eval(clicked_id)  
    selected_patent_id = clicked_id["index"]

    # Retrieve the selected patent data from the dataframe
    patent_data = df[df['patent_id'] == selected_patent_id]
    summary_text = patent_data.iloc[0]['summary_text']

    # Use my classify_industry function to classify the summary text
    top_industries, top_similarities = classify_industry(summary_text)

    # Create bar chart for industries to be displayed 
    bar_chart = px.bar(
        x=top_industries,
        y=top_similarities,
        labels={"x": "Industries", "y": "Similarity Score"},
        color=top_industries, 
        height=300,
    )
    bar_chart.update_layout(showlegend=False)

    # Extract and lemmatize the top 10 keywords using KeyBERT
    raw_keywords = kw_model.extract_keywords(summary_text, top_n=10)
    lemmatized_keywords = list(set(lemmatizer.lemmatize(word) for word, _ in raw_keywords))
    top_keywords = lemmatized_keywords[:5]
    keyword_items = [html.Li(keyword) for keyword in top_keywords]

    return (
        summary_text,
        bar_chart,
        keyword_items
    )

if __name__ == "__main__":
    app.run_server(debug=True)

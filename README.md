# Patent Classification and Keyword generation Dashboard

This project involves building a model to classify patents into industry sectors based on their text summaries, using the **PatensView** 2024 dataset. 
The main exploration and evaluation was made in the **expl.ipynb** notebook, and the concepts derived from that work, were simply implemented to a Dash-plotly application.

<img src="https://github.com/alexdalboel/DS821_exam/blob/main/patentvid.gif" width="700"/>


## Project Overview


- **Preprocessing and evaluation of method**: In expl.ipynb i split the dataset into a smaller size for this project. I conduct some exploratory analysis and extract relevant information from the raw summaries, which contains a lot of 'legalese' language. Only  'Background' and 'Technical backkground' is extracted, and used for the classification task. Furthermore, I evaluate the chosen method of classifying through cosine_similarity by adopting a "silhouette score" method. I compute a silhouette score based on the summaries average similairity(cohesion) with the top-5 classified industries and the the remaining(separation) industry titles.
---
- **Dataset**: The project uses the 2024 Brief Summary Text dataset from PatensView.org, containing patent summaries. This dataset lacks predefined labels for industry sectors, but sections like "TECHNICAL FIELD" within the summaries provide clues for classification.
---
- **Model**: I use the **paraphrase-MiniLM-L6-v2** model to create embeddings in order to classify a summary based on its cosine similarity with a predefined industry description. I also use **keyBERT** model for extracting keywords from the patent summary.
---
- **Dashboard**: A Dash web app will allow users to:
  - Select patents and view their text summaries.
  - Check the applicability of a patent to various industry sectors through a bar chart.
---
## Features

- **Patent Summary Display**: A scrollable area displaying the selected patent's summary text.
- **Industry Classification**: Classify patents into relevant industries based on their summaries.
- **Visualization**: Bar charts showing the applicability of a patent to different industries.
---
## Installation to run dash app locally

1. Clone this repository:
   ```bash
   git clone https://github.com/alexdalboel/DS821_exam.git
   cd DS821_exam
2. (Recommended) Create and activate virtual environment 
- Windows
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
- Mac
  ```bash
  python3 -m venv venv
  source venv/bin/activate

3. Install dependencies 
   ```bash
   pip install -r requirements.txt

4. Run Dash app
   ```bash
   python app.py

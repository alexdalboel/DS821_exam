# Patent Classification and Keyword generation Dashboard

This project involves building a model to classify patents into industry sectors based on their text summaries, using the **PatensView** 2024 dataset. .

<img src="https://github.com/alexdalboel/DS821_exam/blob/main/patentvid.gif" width="700"/>


## Project Overview

- **Dataset**: The project uses the 2024 Brief Summary Text dataset from PatensView.org, containing patent summaries. This dataset lacks predefined labels for industry sectors, but sections like "TECHNICAL FIELD" within the summaries provide clues for classification.
- **Model**: I use the **paraphrase-MiniLM-L6-v2** model to create embeddings in order to classify a summary based on its cosine similarity with a predefined industry description. I also use **keyBERT** model for extracting keywords from the patent summary.
- **Dashboard**: A Dash web app will allow users to:
  - Select patents and view their text summaries.
  - Check the applicability of a patent to various industry sectors through a bar chart.

## Features

- **Patent Summary Display**: A scrollable area displaying the selected patent's summary text.
- **Industry Classification**: Classify patents into relevant industries based on their summaries.
- **Visualization**: Bar charts showing the applicability of a patent to different industries.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/alexdalboel/DS821_exam.git

2. Install dependencies 
   ```bash
   pip install -r requirements.txt

3. Run Dash app
   ```bash
   python app.py

This project performs an end-to-end NLP analysis of AI-related news articles to understand how artificial intelligence is impacting different industries and sectors. It covers the complete data science lifecycle, from data cleaning and entity extraction to topic modeling, sentiment analysis, and macro-trend visualization.

### Project Overview
The repository is organized into five sequential phases:

* **Phase 1: Exploratory Data Analysis (EDA)**
    Performs initial data loading and cleaning of a large news dataset (approx. 200,000 records). Key tasks include language detection, removing duplicate content, and filtering articles to focus specifically on those relevant to AI and its industrial applications.
* **Phase 2: Entity Extraction**
    Utilizes `spaCy` (specifically the `en_core_web_trf` transformer model) and custom `EntityRuler` patterns to identify key organizations, specific AI technologies (e.g., LLMs, Computer Vision), and target industries (e.g., Healthcare, Finance, Manufacturing). This phase identifies which entities are most frequently associated with AI advancements.
* **Phase 3: Topic Modeling**
    Employs `BERTopic` alongside `SentenceTransformers` (`all-MiniLM-L6-v2`) and `UMAP` dimensionality reduction to discover latent themes within the corpus. It categorizes articles into distinct topics such as cybersecurity, medical AI, and financial markets.
* **Phase 4: Sentiment Analysis**
    Features the fine-tuning of a `BERT` foundation model (`bert-base-uncased`) on financial news datasets to classify article sentiment as Positive, Neutral, or Negative. The resulting custom model is used to gauge public and professional perception of AI's impact across different sectors.
* **Phase 5: Macro Trajectory Visualization**
    Synthesizes results from previous phases to create temporal and categorical visualizations. It analyzes how the volume and sentiment of AI-related discussions have evolved over time and identifies which specific AI "impact mechanisms" (like automation or augmentation) are most prevalent in different sectors.

### Technical Stack
* **Language**: Python
* **NLP & ML**: spaCy (Transformers), BERTopic, Hugging Face Transformers, Sentence-Transformers, scikit-learn
* **Data Processing**: Pandas, PyArrow (Parquet), NumPy
* **Visualization**: Matplotlib, Seaborn
* **Infrastructure**: Designed for execution in Google Colab with GPU acceleration (T4)

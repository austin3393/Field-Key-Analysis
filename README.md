# 🔑 Field Key Analysis

This repository contains code and experiments for analyzing **field keys** and their alignment with associated **titles**.  
The goal is to detect and quantify issues such as mismatches, missing context, token validity, and other facets that affect downstream LLM outputs.

---

## 📂 Project Structure
```
field_keys_analysis.ipynb   # Jupyter Notebook with EDA and scoring
requirements.txt            # Python dependencies
```

---

## 🚀 Features
- **Normalization & Tokenization**: Clean and split field keys into tokens.
- **Word Validity Scoring**: Frequency and domain-based validity checks.
- **Containment & Overlap Metrics**:  
  - Jaccard similarity  
  - Containment (key vs. title)  
  - Overlap coefficient  
- **Semantic Scoring**: Cosine similarity with embeddings + Natural Language Inference (NLI) axis integration.
- **Facet Detection**: Flags issues like partial mismatch, temporal context, or invalid tokens.
- **Visualization**: Histograms, pie charts, and summary statistics to track field key quality.

---

## ⚙️ Setup

* 1. Clone the repo
* 2. Create and activate a virtual environment
* 3. Install dependencies
with 
```
pip install -r requirements.txt
```


## 📌 Next Steps

* Validate Key Analysis Pipeline with End-User data such as Progress note feedback, NPS scores, CSAT scores, etc.
* Expand lexicons (temporal, agent, domain-specific terms).
* Create golden dataset to train machine learning model to tune tresholds for facet classification
* Integrate into a pipeline for continuous monitoring of LLM outputs.

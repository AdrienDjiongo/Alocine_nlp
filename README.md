# 🎬 Analyse de Sentiment — Reviews Allociné (NLP)

Projet de classification de sentiment sur 160 000 critiques de films françaises, comparant une approche **ML classique (TF-IDF)** à une approche **Transformers (CamemBERT fine-tuné)**.

## 📊 Résultats

| Modèle                       | Accuracy (Test) | F1-score  |
| ---------------------------- | --------------- | --------- |
| TF-IDF + Logistic Regression | 93.53%          | 0.94      |
| TF-IDF + SVM (LinearSVC)     | 93.12%          | 0.93      |
| **CamemBERT fine-tuné**      | **95.7%**       | **0.957** |

## 🗂️ Dataset

[Allociné Dataset](https://huggingface.co/datasets/tblard/allocine) — 160k critiques de films (2006-2020), équilibrées entre avis positifs et négatifs.

## 🔍 Méthodologie

**1. EDA**

- Distribution des labels, longueur des reviews, WordClouds (avant/après nettoyage)

**2. Préprocessing**

- Nettoyage, suppression stopwords français (NLTK), normalisation

**3. Modèles classiques**

- Vectorisation TF-IDF (unigrammes + bigrammes)
- Logistic Regression et SVM linéaire

**4. Fine-tuning CamemBERT**

- Modèle pré-entraîné `camembert-base` fine-tuné sur 10k échantillons équilibrés
- 3 epochs, GPU T4 (Google Colab)

## 📁 Structure du projet

```
notebooks/
├── LR_&_SVM.ipynb       # EDA + préprocessing + modèles classiques
└── Camambert.ipynb      # Fine-tuning CamemBERT (Google Colab)
outputs/
└── figures/             # Visualisations générées
```

## 🛠️ Stack technique

`Python` `pandas` `scikit-learn` `NLTK` `Transformers (HuggingFace)` `PyTorch` `Google Colab`

## 🚀 Reproduire le projet

```bash
pip install -r requirements.txt
```

Puis exécuter les notebooks dans l'ordre : `LR_&_SVM.ipynb` → `Camambert.ipynb` (sur Colab avec GPU).

## 📈 Conclusion

L'approche Transformers (CamemBERT) apporte un gain de **+2.2 points** par rapport à la meilleure baseline classique, confirmant l'avantage de la compréhension contextuelle face à une représentation purement fréquentielle (TF-IDF) — au prix d'un coût computationnel et d'une complexité d'infrastructure plus élevés.

# Multimodal House Price Valuation (Satellite + Tabular Data)

This project predicts house prices using a **multimodal machine learning approach**, combining:

- **Tabular features** (size, rooms, location, etc.)
- **Satellite images** (neighborhood and environmental context from Sentinel-2)

Beyond prediction accuracy, the project also focuses on **interpretability**, using **Grad-CAM** to understand *what the model looks at* in satellite images when estimating house prices.

---

## Repository Structure

- `data_fetcher.py` – Sentinel Hub image fetching pipeline (lat/long → image tiles + metadata)
- `preprocessing.ipynb` – Data cleaning, log transformation, EDA, and train/validation splits
- `model_training1.ipynb` – Tabular models, CNN feature extraction, multimodal fusion, evaluation
- `explainability.ipynb` – Grad-CAM visualizations and economic interpretation
- `final_predictions.csv` – Final test predictions

### Data Directories
- `data/raw/` – Raw housing dataset (user provided)
- `data/processed/` – Cleaned train/val/test CSVs
- `data/images/` – Downloaded satellite image tiles
- `data/satellite/` – Image metadata (ID → image path)
- `data/embeddings/` – Cached ResNet image embeddings

---

## Methodology Overview

1. **Data Preprocessing**
   - Cleaned raw housing data
   - Handled missing values
   - Applied **log transformation** to house prices to reduce skew and stabilize training
   - Created train/validation splits

2. **Satellite Image Collection**
   - Downloaded Sentinel-2 tiles using latitude/longitude
   - Stored images and metadata locally

3. **Image Feature Extraction**
   - Used **ResNet18 (pretrained on ImageNet)**
   - Extracted **512-dimensional embeddings** per image
   - Cached embeddings for faster experimentation

4. **Multimodal Fusion**
   - Combined scaled tabular features with image embeddings
   - Trained an **XGBoost regressor** on fused features

5. **Evaluation**
   - Metrics evaluated in **log-price space**
   - Converted back to price scale for interpretability (RMSE)

6. **Explainability**
   - Used **Grad-CAM** on an image-only model
   - Visualized important regions for:
     - High-priced houses
     - Low-priced houses
   - Verified that attention maps align with economic intuition
     (density, greenery, roads, water proximity)

---

## Results

- **Fusion Model (Tabular + Image)**
  - R² (log price): ~0.86
  - RMSE (price scale): ~128k

- **Key Insight**
  - Tabular features remain the strongest predictors
  - Satellite images add **contextual signals**, but benefits depend on image quality and coverage
  - Grad-CAM confirms the model focuses on **neighborhood structure**, not random noise

---
## Data Requirements and Setup

### Environment Setup
Python ≥ 3.9 recommended.

```bash
python -m venv .venv
source .venv/bin/activate

---

## How to Run

1. Install dependencies  
   ```bash
   pip install -r requirements.txt

2. Download satellite images 
   ```bash
   python data_fetcher.py
   
3. Run notebooks in order:      
  - `preprocessing.ipynb`
  - `model_training1.ipynb`
  - `explainability.ipynb`

4. Final predictions will be saved as:
   - `final_predictions.csv`

---

## Conceptual Overview

House prices are influenced not only by the physical attributes of a house (such as size, number of rooms, or age), but also by the **surrounding neighborhood** — density, greenery, road access, and overall urban structure.

Traditional machine learning models rely only on **tabular data**, which captures *what the house is*, but often misses *where the house is* and *what surrounds it*.

This project addresses that gap using a **multimodal learning approach**.

---

### 1. Why Multimodal?

We combine two complementary sources of information:

- **Tabular data**  
  Captures structured, well-known drivers of price:
  - square footage  
  - number of bedrooms and bathrooms  
  - location coordinates  
  - construction year, grade, condition  

- **Satellite imagery (Sentinel-2)**  
  Captures visual context:
  - neighborhood density  
  - greenery vs concrete  
  - road networks  
  - proximity to water or open land  

Each modality answers a different question:
- Tabular data → *What is the house?*
- Satellite images → *What does the surrounding area look like?*

---

### 2. How Images Become Numbers

Raw images cannot be directly used by classical ML models.

To solve this:
- A **pretrained ResNet18** CNN is used as a feature extractor
- Each satellite image is converted into a **512-dimensional embedding**
- These embeddings summarize visual patterns such as:
  - texture
  - density
  - spatial layout

The CNN is **not trained from scratch** — it transfers learned visual knowledge from ImageNet to satellite imagery.

---

### 3. Why Log Price Transformation?

House prices are highly skewed:
- Most houses are mid-priced
- A few very expensive houses dominate the scale

We apply a **log transformation** to prices to:
- reduce the impact of extreme outliers
- stabilize training
- allow the model to focus on *relative differences* rather than absolute price gaps

Predictions are converted back to the original price scale for evaluation.

---

### 4. Fusion Strategy

Instead of forcing a single deep network:
- Tabular features are scaled and cleaned
- Image embeddings are concatenated with tabular features
- A **tree-based model (XGBoost)** learns interactions between:
  - structured numeric features
  - visual neighborhood signals

This keeps the system:
- flexible
- interpretable
- robust on limited data

---


   
   

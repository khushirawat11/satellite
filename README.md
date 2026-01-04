# Multimodal House Price Valuation (Satellite + Tabular Data)

This project predicts house prices using a **multimodal machine learning approach**, combining:

- **Tabular features** (size, rooms, location, etc.)
- **Satellite images** (neighborhood and environmental context from Sentinel-2)

Beyond prediction accuracy, the project also focuses on **interpretability**, using **Grad-CAM** to understand *what the model looks at* in satellite images when estimating house prices.

---

## Project Structure

- `data_fetcher.py` – Sentinel Hub image fetching pipeline  
  *(latitude/longitude → bounding box → satellite tile saved locally + metadata).*

- `preprocessing.ipynb` –  
  Data cleaning, feature engineering, log-price normalization, and creation of train/validation splits.

- `model_training1.ipynb` –  
  Tabular model training, ResNet18 image embedding extraction, multimodal (tabular + image) fusion using XGBoost, evaluation, and test prediction generation.

- `explainability.ipynb` –  
  Model interpretability using **Grad-CAM** to visualize which regions of satellite images influence high and low house price predictions.

- `final_predictions.csv` –  
  Final submission-ready CSV containing predicted house prices for the test dataset.

---

## Data Directory

- `data/`
  - `data/images/` – Downloaded satellite image tiles (`.png`)
  - `data/satellite/image_metadata.csv` – Mapping between house `id` and satellite image paths with download status
  - `data/processed/` – Cleaned and processed datasets:
    - `train_final.csv`
    - `val_final.csv`
    - `test2(test(1)).csv`
  - `data/embeddings/` – Cached CNN embeddings:
    - `resnet18_train_embeddings.csv`
    - `resnet18_val_embeddings.csv`
    - `resnet18_test_embeddings.csv`

- `train(1)(train(1)).csv` – Original raw housing dataset (user-provided)

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

### 5. Why Explainability Matters

Accuracy alone is not sufficient.

We use **Grad-CAM** on an image-only model to verify:
- whether the model focuses on **economically meaningful regions**
- not on random noise or image borders

By comparing:
- high-priced houses
- low-priced houses

we check whether attention maps highlight:
- dense urban zones
- greenery
- road connectivity
- open vs congested spaces

Models whose attention appears random are rejected.

---

### 6. Key Insight

In this dataset:
- **Tabular features remain the strongest predictors**
- Satellite imagery adds **context**, not dominance
- Visual signals are more useful for:
  - validating model behavior
  - understanding neighborhood effects
  - supporting economic intuition

This confirms an important lesson:
> Multimodal learning improves *understanding* even when it does not dramatically improve *metrics*.

---

### 7. Final Takeaway

This project demonstrates how:
- multimodal ML can bridge structured data and spatial context
- explainability tools can validate economic reasoning
- careful modeling choices often matter more than architectural complexity

The result is a model that is not only predictive, but **trustworthy and interpretable**.


   
   

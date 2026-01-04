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

## 3. Data Requirements and Setup

1. **Create a virtual environment** (recommended, Python ≥ 3.9):

```bash
python -m venv .venv
source .venv/bin/activate
```
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```
3. **Place the King County dataset**:

- Download the King County housing CSV (commonly named `kc_house_data.csv`) and save it to:
  - `data/raw/kc_house_data.csv`

4. **Configure Sentinel Hub credentials**:

- Create a `.env` file in the project root or set environment variables (recommended):
  - `SENTINELHUB_CLIENT_ID`
  - `SENTINELHUB_CLIENT_SECRET`
  - Optionally, additional config such as:
    - `SENTINELHUB_INSTANCE_ID` (for older setups)
    - Default collection / resolution if needed.

The `data_fetcher.py` module will read these environment variables and handle authentication.

---
## 4. Sentinel Hub Image Fetching (Engineering Considerations)

### Why Resolution and Tile Size Matter Economically

**Zoom / Ground Sampling Distance (GSD):**

- Too coarse (60–100 m/pixel): individual properties blur together and neighborhood structure is lost.
- Too fine (sub-meter): redundant details, heavy downloads, and higher risk of overfitting.
- This project uses a **parcel-scale context window** (≈10 m/pixel, ~224×224 px) to capture:
  - Greenery and tree canopy
  - Road layout and accessibility
  - Proximity to water bodies
  - Density of surrounding buildings

**Bounding Box Size:**

- Buyers value **nearby context**, not just the parcel itself.
- A fixed-radius bounding box (≈250–500 m) centered on each property encodes **neighborhood quality**, including amenities and disamenities.

**Determinism & Reproducibility:**

- Image filenames are deterministic functions of the property `id`.
- Metadata linking each `id` to its image path is stored in:
  ```text
  data/satellite/image_metadata.csv
**Robustness engineering**:

- Automatic **retry logic** with exponential backoff for transient Sentinel Hub errors.
- Graceful handling of:
  - Missing tiles or cloud cover (configurable filters).
  - API rate limits (throttling and logging).
  - Partial coverage (e.g., edges of acquisition area).

Details are implemented in `data_fetcher.py`.

---
### 5. Workflow

1. **Fetch Satellite Images**
   
   Run the Sentinel Hub image downloader:
   ```bash
   python data_fetcher.py
   This step:
   ```
   
*   Downloads Sentinel-2 satellite images using latitude/longitude
*   Saves images to `data/images/`
*   Creates `data/satellite/image_metadata.csv` mapping `id → image_path`

2.  **Preprocessing & Exploratory Data Analysis**

   Open `preprocessing.ipynb`:
   *   Cleans tabular housing data
*   Handles missing values and outliers
    
*   Applies **log price normalization**
    
*   Performs tabular EDA (distributions, correlations)
    
*   Performs geospatial EDA (price vs latitude/longitude)
    
*   Creates leakage-aware train/validation splits
    
*   Outputs:
    *   `data/processed/train_final.csv`
    
    *   `data/processed/val_final.csv`
    
    *   `data/processed/test2(test(1)).csv`

3. ***Model Training (Tabular + Image + Fusion)***

   *   Loads processed tabular data

*   Extracts **ResNet-18 CNN embeddings** from satellite images
    
*   Saves embeddings to:
    
    *   `data/embeddings/resnet18_train_embeddings.csv`
        
    *   `data/embeddings/resnet18_val_embeddings.csv`
        
*   Trains:
    
    *   Tabular-only models
        
    *   Image-only models
        
    *   **Multimodal fusion model (XGBoost)**
        
*   Evaluates using:
    
    *   RMSE
        
    *   R²
        

*   Selects the best-performing mode

4. Open `notebooks/explainability.ipynb`:
   - Run Grad-CAM (or similar) on the image model.
   - Inspect where the model focuses for **high vs. low priced** houses.
   - Reject models whose attention is spatially random or economically uninterpretable.
  
5. **Generate test predictions**
   *   Ensure test data exists at:
    
*   `data/processed/test2(test(1)).csv`
    

*   Run:

*   `python predict_test_prices.py`
    

*   This step:

*   Loads test data
    
*   Matches available satellite images
    
*   Extracts CNN embeddings
    
*   Applies the trained fusion model
    
*   Generates predictions for **all test rows**
    
*   Saves output to:
    
    `final_predictions.csv`
    

*   Format:

*   `id, predicted_price`

   

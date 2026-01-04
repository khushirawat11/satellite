##Satellite Imagery-Based Property Valuation

This project predicts house prices using a **multimodal approach**, combining:
- **Tabular features** (size, rooms, location, etc.)
- **Satellite images** (neighborhood context from Sentinel-2)

The goal is not only prediction accuracy, but also **interpretability** using Grad-CAM to understand *what the model looks at* in satellite images.

---

### 1. Repository Structure
CDC/
│
├── data/
│   ├── images/                          # Downloaded satellite images (.png)
│   ├── satellite/
│   │   └── image_metadata.csv           # id → image_path mapping, download status
│   ├── processed/
│   │   ├── train_final.csv              # Cleaned training data
│   │   ├── val_final.csv                # Validation split
│   │   └── test2(test(1)).csv           # Test data (no labels)
│   └── embeddings/
│       ├── resnet18_train_embeddings.csv  # CNN embeddings for train images
│       ├── resnet18_val_embeddings.csv    # CNN embeddings for validation images
│       └── resnet18_test_embeddings.csv   # CNN embeddings for test images
│
├── data_fetcher.py                      # Sentinel Hub image fetching pipeline
│                                       # (lat/long → bounding box → satellite tile)
│
├── preprocessing.ipynb                  # Data cleaning, feature engineering,
│                                       # log normalization, train/val split
│
├── model_training1.ipynb                # Tabular models, image embedding extraction,
│                                       # multimodal fusion (XGBoost),
│                                       # and final price predictions
│
├── explainability.ipynb                 # Grad-CAM visualizations to interpret
│                                       # what the image model focuses on
│
├── final_predictions.csv                # Final test-set price predictions
│
├── train(1)(train(1)).csv               # Original raw dataset (user-provided)
│
├── .gitignore                           # Git ignore rules
├── README.md                            # Project documentation

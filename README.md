# 🦠 Machine Learning Predictive Architecture for Harmful Algal Blooms (Lake Erie Basin)
An independent data science research framework implementing tree-based classification pipelines to map ambient hydroclimate drivers against microcystin ecological toxicity hazards.

## 🚀 Interactive Deployment Link
Click the link below to open the complete source script and launch the interactive full-stack simulation dashboard inside a live cloud execution container:

**[👉 Launch Live Interactive Model Dashboard](https://423e252ed73ca06e5b.gradio.live))**

*(Note: Adjust the water metric parameters via the step-calibrated sliders to observe real-time variance shifts in the Random Forest classification vector).*

## 🔬 Scientific Abstract & Problem Statement
Harmful Algal Blooms (HABs) driven by cyanobacteria yield microcystins that degrade aquatic ecosystems and threaten municipal water grids. This project utilizes raw observational monitoring arrays from the EPA ScienceHub repository to model bloom conditions based on ambient drivers (Temperature, Dissolved Oxygen, and Nutrient Indicators) rather than biomass symptoms (Chlorophyll-a).

### Methodology & Technical Implementation:
* **Feature Insulation:** Biomass indicators are stripped from the training matrices to prevent data leakage, forcing models to rely strictly on predictive physical drivers.
* **Target Discretization:** Raw index concentrations are mapped directly into classification boundaries corresponding to EPA recreational water safety advisory limits.
* **Comparative Benchmarking:** An advanced tree-based `RandomForestClassifier` is trained against a statistical `LogisticRegression` baseline using stratified train/test partitions to maintain class integrity across chronological blocks.

## 📊 Model Performance Analysis
The predictive pipeline evaluates classification parameters across multiple validation vectors:
* **Baseline Statistical Standard (Logistic Regression) Accuracy:** `0.782`
* **Optimized Tree Classifier (Random Forest) Accuracy:** `0.894`
* **Random Forest Macro F1-Score:** `0.887`

*A complete Gini variable importance matrix is computed within the pipeline, revealing that water temperature thresholds dictate the primary structural variance paths in prediction runs.*

## 🛠️ Research Engineering Stack
* **Core Analytics:** Python 3, `pandas`, `numpy`
* **Machine Learning Pipeline:** `scikit-learn` (`RandomForestClassifier`, `LogisticRegression`)
* **Data Visualization:** `matplotlib`
* **UI Deployment Prototyping:** `gradio`

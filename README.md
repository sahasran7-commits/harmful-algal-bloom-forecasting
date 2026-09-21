# harmful-algal-bloom-forecasting
# 🌊 Multi-Criteria Decision Optimization Framework for PFAS Chemical Remediation
An independent data engineering project utilizing direction-specific Min-Max normalization matrices to evaluate the thermodynamic limits and engineering viability of "Forever Chemical" degradation pathways.

## 🚀 Interactive Deployment Link
Click the link below to open the complete source script and launch the interactive full-stack simulation dashboard inside a live cloud execution container:

**[👉 Launch Live Interactive Multi-Criteria Slider Dashboard](YOUR_GRADIO_OR_HUGGINGFACE_URL_HERE)**

*(Note: Adjust the allocation sliders via the step-calibrated increments to observe real-time variance shifts in the multi-scenario decision matrix).*

## 🔬 Scientific Abstract & Problem Statement
Per- and polyfluoroalkyl substances (PFAS) are anthropogenic chemical compounds defined by exceptionally strong Carbon-Fluorine (C-F) bonds, resulting in severe environmental bioaccumulation. This project implements a Multi-Criteria Decision Analysis (MCDA) framework to evaluate competing treatment strategies (Hydrated Electron Radical Processing, Electrochemical Membrane Oxidation, Granular Activated Carbon, and Direct Thermal Cleaving) under conflicting operating priorities.

### Methodology & Technical Implementation:
* **Constraint Weight Normalization:** Saliency inputs are strictly forced to sum to 1.00 to guarantee a mathematically coherent optimization engine across parameters.
* **Direction-Specific Feature Scaling:** Applies Benefit-Maximization equations to recovery and kinetics, and Cost-Minimization equations to energy and operational costs. This bounds conflicting raw units safely between an enclosed envelope (0.0 to 1.0) to eliminate metric scale bias.
* **Parameter-Space Sensitivity Analysis:** Evaluates four distinct predefined baseline engineering priority matrices (Energy Focus, Velocity Focus, Mineralization Focus, and Capex Focus) to calculate structural tipping points in recommendation pathways.

## 📊 Analytical Scoring Architecture
The recommendation system uses a normalized weight-assignment engine to score chemical processing pathways based on conflicting engineering trade-offs:

\[\text{Viability Score } (S) = (w_R \cdot R^*) + (w_K \cdot K^*) + (w_E \cdot E^*) + (w_C \cdot C^*)\]

\[\text{Where } w_R + w_K + w_E + w_C = 1.0\]

## 🛠️ Research Engineering Stack
* **Core Analytics:** Python 3, `pandas`, `numpy`
* **Mathematical Visualization:** `matplotlib`
* **UI Deployment Prototyping:** `gradio`

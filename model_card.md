# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier designed to predict whether an individual's annual income exceeds $50,000 based on demographic and employment attributes extracted from Census Bureau data. It was developed in July 2026 as part of a Machine Learning DevOps deployment project. The architecture utilizes the scikit-learn implementation of a Random Forest with default hyperparameters and a fixed random state of 42 to ensure consistent reproducibility.
## Intended Use
The main purpose of this model is to function as a predictive tool for economic and demographic research. It enables users to examine the relationship between personal characteristics and income levels. The model is accessible through a RESTful FastAPI service for real-time inference. It is designed for educational and exploratory analytical tasks and should not be used for automated financial profiling, credit scoring, or making legally binding decisions related to hiring or lending.
## Training Data
The model was trained on a public dataset obtained from the Census Bureau, which represents a stratified random sample of historical census records. The dataset includes categorical features such as workclass, education, marital status, occupation, relationship, race, sex, and native country. An 80% random split of the original data was used for training. Categorical attributes were processed using a one-hot encoder, and the target income labels were binarized before training.
## Evaluation Data
The evaluation was conducted on a held-out test subset that comprised the remaining 20% of the original Census Bureau dataset. This dataset was completely separate from the training pipeline. It underwent the same categorical transformations using the One-Hot Encoder and Label Binarizer, which were fitted exclusively on the training partition to ensure that no data leakage occurred.
## Metrics
Precision: 0.7419 
Recall: 0.6384 
F1: 0.6863
## Ethical Considerations
This dataset is based on historical census data, which reflects existing socioeconomic disparities related to gender, race, and nationality at the time of collection. For instance, certain demographic groups may show skewed predictions due to historical imbalances in high-paying sectors. It is important for developers and auditors to closely monitor the model's performance across these different demographic groups to identify and address any biases, ensuring that the model does not perpetuate unfair discrimination.
## Caveats and Recommendations
The underlying census data is outdated and may not accurately reflect current economic realities, inflation, shifting job markets, or modern wage distributions. It is recommended that the model be regularly retrained with updated economic indicators. Additionally, hyperparameter tuning could be explored to further optimize the balance between precision and recall, depending on the specific cost of false positives versus false negatives in your deployment use case.
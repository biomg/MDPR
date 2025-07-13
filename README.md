# MDPR
T cell receptor sequences (TCR-seq) reflect the body's immune status, making their accurate detection crucial for cancer diagnosis and treatment. Current prediction methods for cancer-related TCR-seq often focus solely on the sequence structure, neglecting its spatial structure. Therefore, we propose a multimodal deep learning method based on parallel and residual structures (MDPR), for the detection of cancer-related TCR-seq. MDPR can effectively integrate the spatial and sequence structures of TCR-seq for accurate prediction of cancer-related TCR-seq. First, we introduce a TCR-seq coordinate-hot encoding method based on atomic three-dimensional spatial coordinates and atomic one-hot encoding, allowing for more effective extraction of the spatial structural features of TCR-seq. Second, we use high-dimensional word vectors instead of the amino acid feature vectors traditionally used by other researchers. Third, we pretrain the spatial feature extraction module and then conduct joint training with the sequence feature extraction module. This approach allows the model to better consider the relationship between the two modalities, resulting in more accurate predictions. In the end, MDPR achieved an area under the curve (AUC) of 0.971 after ten rounds of three-fold cross-validation on the dataset. The AUC of MDPR is 5% higher than that of the previous best method. In short, we propose an artificial intelligence method called MDPR, and apply it to the biomedical field.

# Dependency:
python 3.7.1 <br>
torch 1.12.1 <br>
numpy 1.21.3 <br>
sklearn 0.23.2 <br>
# Data 
All datasets we used can be found at https://zenodo.org/records/15871815.
# Data processing
If you have your own dataset that needs preprocessing, you can use the functions provided in data_processing.py
# Usage:
python train_and_test.py






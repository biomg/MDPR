# MDPR
Multimodal deep learning method based on parallel and residual structures(MDPR) is a deep learning model developed by us for predicting cancer-related T-cell receptor sequences (TCRs). This deep learning model is designed to effectively extract both the spatial structural features and sequence features of TCRs.

# Dependency:
python 3.7.1 <br>
torch 1.12.1 <br>
numpy 1.21.3 <br>
sklearn 0.23.2 <br>
# Data 
We utilized the dataset provided by Wong et al. as the training and testing dataset for MDPR. This dataset consists of 300,000 TCRs, comprising 100,000 cancer-related TCRs and 200,000 non-cancer TCRs. The original dataset can be found in "TrainingData.rar" or downloaded from https://github.com/cew88/AutoCAT.

To obtain the spatial structure of TCRs, we utilized the ESM-fold protein structure prediction tool to acquire the spatial structural features of all TCRs in the original dataset. We preprocessed the spatial and sequence information of TCRs using our proposed coordinate-hot encoding method and word vectors, thereby constructing a multi-modal dataset for TCRs. You can find this dataset at https://pan.baidu.com/s/1KkCzP5o2ADY928RTkeo5VQ,password is "9eej".

Users can also use "data_processing.py" to process their own TCRs data and build their own TCRs dataset. "data_processing.py" includes three parts: TCRs spatial structure prediction, coordinate-hot encoding, and amino acid sequence preprocessing.

# Usage:

python train_and_test.py

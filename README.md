# CT-RBP
This repository contains the implementation of CT-RBP, which is a model based on CNN and Transformer, used to predict the interactions between RNA and RBP.

# 1. Dependencies
This model is designed to operate in the following environment:
```
GPU: RTX 2080 Ti
CPU: 12 vCPU Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
CUDA Version: 11.0
Python: 3.8
Ubuntu: 18.04
```

# 2. Install requirements
Install the required packages by running:
```
git clone https://github.com/Kaylee-Zhou/CT-RBP
cd CT-RBP
python3 -m pip install -r requirements.txt
```

# 3. Run the model
To run this model, the following steps need to be followed:
```
  Step 1: Suppose your dataset is named "xxx". Place the prepared datasets xxx.negative.fa and xxx.positive.fa into the "xxx" folder.
  Step 2: Run the "preprocessing.ipynb" file to obtain the preprocessed dataset (replace "HNRNPC" in the line "dataset_path = "HNRNPC"" with "xxx").
  Step 3: Run the "CT-RBP.ipynb" file to perform training and prediction in one step(replace 'HNRNPC' in the line "data_path = 'HNRNPC'" with "xxx").
```
Note: The output file will be located in "xxx"/Output.

# 4. Dataset visualization
To view the information of the visualized datasets, you can:
```
Run the three ".ipynb" files in the Visualization folder (In the line "df = pd.read_csv('../HNRNPC/eval.tsv', sep='\t')", the "HNRNPC" should be replaced with "xxx").
```
Note: The output file will be located in "Visualization/PNG".

# 5. Performance
This model achieves the following performance on the HNRNPC and U2AF2 datasets:
```
【HNRNPC】
AUROC:       0.8759
AUPRC:       0.8911
Accuracy:    0.8133
Precision:   0.8389
Sensitivity: 0.7962
F1-Score:    0.8170
Specificity: 0.8322
MCC:         0.6277

【U2AF2】
AUROC:       0.8509
AUPRC:       0.8721
Accuracy:    0.7800
Precision:   0.7791
Sensitivity: 0.8089
F1-Score:    0.7938
Specificity: 0.7483
MCC:         0.5587
```
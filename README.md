# CT-RBP
CT-RBP is a model based on convolutional neural network (CNN) and transformer, used to predict the interaction between RNA and RBP.

This repository contains:

1. Preprocessing.ipynb and Preprocessing_Dataset2.ipynb: The codes for preprocessing the .fa file data into the .tsv format.
2. Visualization folder: The codes for visualizing the .tsv files.
3. CT-RBP.ipynb and CT-RBP_Dataset2.ipynb: The implementation code of the CT-RBP model and the SHAP analysis code.
4. RAG.ipynb: The code for analyzing model results  and providing experimental suggestions using advanced retrieval generation technology.
5. GUI folder: A GUI website application that integrates model prediction and RAG functions.
6. Document folder: The documentations of this project.

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
conda install -c bioconda viennarna
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
This model achieves the following performance on the HNRNPC dataset:

```
AUROC:       0.8626
AUPRC:       0.8826
Accuracy:    0.7933
Precision:   0.8146
Sensitivity: 0.7834
F1-Score:    0.7987
Specificity: 0.8042
MCC:         0.5870
```

On the U2AF2 dataset:

```
AUROC:       0.8332
AUPRC:       0.8580
Accuracy:    0.7533
Precision:   0.7546
Sensitivity: 0.7834
F1-Score:    0.7687
Specificity: 0.7203
MCC:         0.5051
```

Note: More details please check at "CT-RBP.ipynb" and "CT-RBP_Dataset2.ipynb".

# 6. Run RAG file

```
Open the "Rag.ipynb" file and replace the last block code: test_input = """""" to your own.
```

Note: Remember to upload the API_KEY if it does not work.

# 7. Run GUI website application

```
Open the GUI folder as project. Select the "app.py" file and run it.
```

Note: Remember to upload the API_KEY if it does not work.

# 8. Documents

```
All the documents should be submmitted are stored in the "Document" folder.
```

# 9. Others

```
The datasets used in these project are from: http://www.csbio.sjtu.edu.cn/bioinf/RBPsuite/dataset_new.html#start
```


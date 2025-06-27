# PAAD_Fibrosis

Pancreatic cancer has a poor prognosis (5-year survival: 12%), with >80% of patients diagnosed at advanced stages where chemotherapy is the primary treatment. While tumor fibrosis significantly impacts chemotherapy efficacy in pancreatic ductal adenocarcinoma (PDAC), its prognostic role remains unclear. This study developed an AI-based model for non-invasive fibrosis prediction using a three-phase design: (1) Deep learning quantification of stromal area in whole-slide histopathology images to establish fibrosis grading; (2) Construction of a CT radiomics model for preoperative fibrosis prediction; (3) Investigation of associations between fibrosis severity and chemotherapy response across multiple cohorts.

## Experiment 1: Tissue Identification Model

A deep learning model was developed to automatically quantify eight tissue subtypes in whole-slide images (WSIs) from three cohorts. The model achieved classification accuracies of 96.00% , 95.81% , and 97.12% , with activation value-weighted fusion generating precise segmentation maps. Patients were stratified into high/low-fibrosis groups using the median stromal proportion. High fibrosis correlated with significantly improved overall survival in all cohorts (log-rank p-values: 0.0338, 0.0318, 0.0196). Transcriptomic analysis of 51 patients confirmed enrichment of fibrosis-associated genes, validated by GO/KEGG pathway analyses across biological processes.

## Experiment 2: Fibrosis Prediction Model & Survival Outcomes

A CT radiomics model trained on training set achieved mean five-fold cross-validation AUC of 0.736, maintaining 0.718 AUC in external validation . Applying this non-invasive fibrosis assessment to 295 chemotherapy-treated PDAC patients revealed key survival associations: High-fibrosis patients receiving AG chemotherapy showed significantly improved overall survival (log-rank p=0.0018) and progression-free survival (p=0.0371). No survival differences based on fibrosis level were observed for FOLFIRINOX or SOXIRI regimens. This demonstrates that AI-powered CT-based fibrosis prediction can identify patients most likely to benefit from specific chemotherapy regimens, particularly AG for high-fibrosis cases.



## How to run the code?

The code is written in Python, MATLAB and R. For the two abovementioned experiments, run the scripts in the corresponding folders sequentially according to the file names. 



## Contact information

If you have any questions, feel free to contact me.

Yulong Han

Shenzhen University, Shenzhen, China
E-mail: hanyl0805@163.com
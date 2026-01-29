This project implements a complete end‑to‑end Named Entity Recognition (NER) pipeline for extracting job titles and skills from raw job descriptions. The workflow includes building a structured corpus, generating IOB‑tagged annotations, preprocessing text with a BERT tokenizer, and encoding labels for model training. All tokenized inputs are saved as pickle files, while label sequences and mappings are stored in NumPy and JSON formats for easy reuse. The training module fine‑tunes a BERT‑base model for token classification, using PyTorch and Hugging Face Transformers. 
The pipeline is modular, reproducible, and designed to support future evaluation, inference, and explainability steps. This forms the foundation for a robust job‑information extraction system


              precision    recall  f1-score   support
   JOB_TITLE       1.00      1.00      1.00       315
       SKILL       1.00      1.00      1.00      5202

   micro avg       1.00      1.00      1.00      5517
   macro avg       1.00      1.00      1.00      5517
weighted avg       1.00      1.00      1.00      5517

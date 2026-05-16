(Disclaimer: no change has been made to code. only the correct one has been uploaded over the earlier version. sorry for the late realization.)
# Prior Case Retrieval with Rhetorically Annotated LegalSeg Data

> A two-phase NLP pipeline for Prior Case Retrieval (PCR) on IL-PCR data, leveraging rhetorical segmentation via a Hier-BiLSTM CRF model trained on the LegalSeg corpus.

---

## Overview

This project tackles **Prior Case Retrieval (PCR)** — the task of finding legally relevant precedent cases given a query document. The core idea is to use **rhetorical segmentation** (labeling functional parts of a legal document) to improve retrieval quality.

The project is divided into two phases:

**Phase I — Reproducing Baseline Results**
Reproduces the results from the *Segment First, Retrieve Better* paper, which performs PCR on IL-PCR data using a pre-trained Hier-BiLSTM CRF model with 7 rhetorical labels. 

{Facts, Ruling by Lower Court, Argument, Statute, Precedent, Ratio of the decision, Ruling by Present Court}.

**Phase II — Training on LegalSeg & Cross-Corpus Inference**
Trains the Hier-BiLSTM CRF model from scratch on the richer **LegalSeg** corpus, runs inference on IL-PCR queries and documents to produce new rhetorical annotations, and then performs PCR using the segment-first approach with these new annotations.

{Facts, Issue, Arguments of Petitioner,
Arguments of Respondent, Reasoning,
Decision, None}.

---

## Key Components

| Component | Description |
|---|---|
| **Hier-BiLSTM CRF** | Hierarchical BiLSTM-CRF model used for rhetorical role labeling of legal text |
| **LegalSeg** | Richly annotated legal corpus with 7 rhetorical labels, used for training in Phase II |
| **IL-PCR** | Indian Legal Prior Case Retrieval dataset — queries and candidate cases |
| **Segment First, Retrieve Better** | Paper whose PCR pipeline this project reproduces and extends |

---

## Repository Structure

```
NLP-PROJECT-PRIOR-CASE-RETRIEVAL/
├── Phase-I/
│   ├── Pipeline/
│   │   └── code.ipynb                  
│   ├── Rhetorical Segmentation/
│   │   ├── annotated_queries/
│   │   │   ├── test_queries.txt
│   │   │   ├── train_queries.txt
│   │   │   └── val_queries.txt
│   │   └── code.ipynb                
│   └── README.md
├── Phase-II/
│   ├── Pipeline/
│   │   └── code.ipynb
│   ├── Rhetorical Segmentation/
│   │   ├── annotated_queries/
│   │   │   ├── test_queries.txt
│   │   │   ├── train_queries.txt
│   │   │   └── val_queries.txt
│   │   └── code.py                    
│   └── README.md
├── Trace Retriever Implementation/
│   ├── CODES.....a few more
└── README.md
```
---

## References

- *Segment First, Retrieve Better: Realistic Legal Search via Rhetorical Role-Based Queries* — prior case retrieval using rhetorical segmentation

@misc{nigam2025segmentfirstretrievebetter,
      title={Segment First, Retrieve Better: Realistic Legal Search via Rhetorical Role-Based Queries}, 
      author={Shubham Kumar Nigam and Tanmay Dubey and Noel Shallum and Arnab Bhattacharya},
      year={2025},
      eprint={2508.00679},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.00679}, 
}
- *LegalSeg: Unlocking the Structure of Indian Legal Judgments Through Rhetorical Role Classification* — rhetorically annotated Indian legal corpus

@misc{nigam2025legalsegunlockingstructureindian,
      title={LegalSeg: Unlocking the Structure of Indian Legal Judgments Through Rhetorical Role Classification}, 
      author={Shubham Kumar Nigam and Tanmay Dubey and Govind Sharma and Noel Shallum and Kripabandhu Ghosh and Arnab Bhattacharya},
      year={2025},
      eprint={2502.05836},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.05836}, 
}
- *Identification of Rhetorical Roles of Sentences in Indian Legal Judgments* — hierarchical sequence labeling model for legal documents

@misc{bhattacharya2019identificationrhetoricalrolessentences,
      title={Identification of Rhetorical Roles of Sentences in Indian Legal Judgments}, 
      author={Paheli Bhattacharya and Shounak Paul and Kripabandhu Ghosh and Saptarshi Ghosh and Adam Wyner},
      year={2019},
      eprint={1911.05405},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/1911.05405}, 
}

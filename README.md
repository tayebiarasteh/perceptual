# Perceptual implications of automatic anonymization in pathological speech



Overview
------

* This is the official repository of the paper [**Perceptual implications of automatic anonymization in pathological speech**](TODO).

Abstract
------
....

### Prerequisites

The software is developed in **Python 3.10**. For the deep learning, the **PyTorch 2.1** framework is used.



Main Python modules required for the software can be installed from ./requirements:

```
$ conda env create -f requirements.yaml
$ conda activate perceptual
```

**Note:** This might take a few minutes.


Code structure
---

Our source code for training and evaluation of the deep neural networks, speech analysis and preprocessing are available here.

* You can run all statistical analyses from *./analyses.py*.


Structure of the **PathologyAnonym** project:

1. Everything can be run from *./PathologyAnonym_main.py*. 
* The data preprocessing parameters, directories, hyper-parameters, and model parameters can be modified from *./PathologyAnonym_main/configs/config.yaml*.
* Also, you should first choose an `experiment` name (if you are starting a new experiment) for training, in which all the evaluation and loss value statistics, tensorboard events, and model & checkpoints will be stored. Furthermore, a `config.yaml` file will be created for each experiment storing all the information needed.
* For testing, just load the experiment which its model you need.

2. The rest of the files:
* *./PathologyAnonym_main/data/* directory contains all the data preprocessing, and loading files.
* *./PathologyAnonym_main/mcAdams_Anonym/* directory contains all the files for anonymization using McAdams coefficient method.
* *./PathologyAnonym_main/PathologyAnonym_Train_Valid.py* contains the training and validation processes.
* *./PathologyAnonym_main/pathanonym_Prediction.py* all the prediction and testing processes.
* For EER calculation you should use either of the anonymization methods' folders based on your need.



------
### In case you use this repository, please cite the original papers:

#### 1:

Tayebi Arasteh S, et al. *Perceptual consequences of automatic anonymization in pathological speech*. ArXiv (2025).


### BibTex

    @article {pathology_perceptual,
      year = {2025},
    }


#### 2:

Tayebi Arasteh S, Arias-Vergara T, Pérez-Toro P, et al. *Addressing challenges in speaker anonymization to maintain utility while ensuring privacy of pathological speech*. Communications Medicine, 4, 182 (2024). DOI: https://doi.org/10.1038/s43856-024-00609-5


### BibTex

    @article {pathology_anonym,
      author={Tayebi Arasteh, Soroosh and Arias-Vergara, Tomás and Pérez-Toro, Paula Andrea and Weise, Tobias and Packhäuser, Kai and Schuster, Maria and Noeth, Elmar and Maier, Andreas and Yang, Seung Hee},
      title = {Addressing challenges in speaker anonymization to maintain utility while ensuring privacy of pathological speech},
      volume={4},
      ISSN={2730-664X},
      url={http://dx.doi.org/10.1038/s43856-024-00609-5},
      DOI={10.1038/s43856-024-00609-5},
      journal={Communications Medicine},
      year = {2024},
    }


#### 3:

Tayebi Arasteh S, Weise T, Schuster M, et al. *The effect of speech pathology on automatic speaker verification: a large-scale study*. Scientific Reports (2023) 13:20476. https://doi.org/10.1038/s41598-023-47711-7



### BibTex

    @article {pathology_asv,
    author = {Tayebi Arasteh, Soroosh and Weise, Tobias, and Schuster, Maria and Noeth, Elmar and Maier, Andreas and Yang, Seung Hee},
    title = {The effect of speech pathology on automatic speaker verification: a large-scale study},
    year = {2023},
    pages = {20476},
    volume = {13},
    doi = {https://doi.org/10.1038/s41598-023-47711-7},
    journal = {Scientific Reports}
    }

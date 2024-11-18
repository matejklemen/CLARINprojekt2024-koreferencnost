This repository serves as a code archive for the deliverables produced within the CLARIN.SI 2024 project "Implementacija podpore za razširjeno uporabo 
slovenskih virov za odkrivanje koreferenčnosti". 
The aim of the project was to enable easier use and broader recognition of Slovene coreference data:
- via a convenient `datasets` library data loading implementation;
- via a conversion into the universal CorefUD data format;
- via a unified benchmarking implementation within the SloBENCH evaluation framework.

## Contents
### Conversion_UDCoref
The folder contains scripts for converting the coref149 and SentiCoref corpora from their original formats into the 
CorefUD CoNLL-U format. For the scripts to work, the following data in raw format needs to be placed within the folder:
- `coref149_v1.0/` = [coref149 corpus](http://hdl.handle.net/11356/1182);
- `SUK.TEI/` = [SentiCoref corpus](http://hdl.handle.net/11356/1959);
- `senticoref_private` = publicly unavailable, intended to be accessible only via the [SloBENCH evaluation framework](https://slobench.cjvt.si/).

Afterward, the corresponding scripts (`convert_coref149.py`, `convert_senticoref.py`, `convert_senticoref_private.py`) can be run successfully.

### Benchmarking_SloBENCH
The folder contains implementation of coreference resolution evaluation within the SloBENCH evaluation framework for the coref149 and SentiCoref corpora.
The code is here for an archival purpose, and the following pull request (and the repository) should be observed for a completely up to date version:
https://github.com/clarinsi/slobench-eval-docker/pull/3.

### DataLoaders_HuggingFace
The folder contains scripts that support user-friendly data loading of the coref149 and SentiCoref corpora within the HuggingFace datasets library.
Additionally, the implementation places the two resources on an [international portal](https://huggingface.co/datasets), potentially giving it more recognition.
The code is here for an archival purpose, and the scripts on the HuggingFace portal should be observed for an up-to-date version:
[cjvt/senticoref](https://huggingface.co/datasets/cjvt/senticoref), [cjvt/coref149](cjvt/coref149).



![CLARIN.SI logo](clarin-logo.png)

The code contained here was produced within the CLARIN.SI 2024 project "Implementacija podpore za razširjeno uporabo 
slovenskih virov za odkrivanje koreferenčnosti".

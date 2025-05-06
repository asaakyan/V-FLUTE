# Understanding Figurative Meaning through Explainable Visual Entailment

This is a repository for the paper [Understanding Figurative Meaning through Explainable Visual Entailment](https://arxiv.org/abs/2405.01474)

# Data

The **V-FLUTE** dataset is available on [HuggingFace](https://huggingface.co/datasets/ColumbiaNLP/V-FLUTE).

# Models

You can reproduce fine-tuned the models using the scripts in LLaVA/scripts/vflute and hyperparameters in the paper.

Our best model is available on HuggingFace here:
[LLaVA-1.5-7b-eViL-VFLUTE-lora](https://huggingface.co/asaakyan/LLaVA-1.5-7b-eViL-VFLUTE-lora)

# Evaluation

See the eval folder for scripts to compute F1@ExplanationScore and and run inference on the test set.

# Citation

```
@inproceedings{saakyan-etal-2025-understanding,
    title = "Understanding Figurative Meaning through Explainable Visual Entailment",
    author = "Saakyan, Arkadiy  and
      Kulkarni, Shreyas  and
      Chakrabarty, Tuhin  and
      Muresan, Smaranda",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.1/",
    pages = "1--23",
    ISBN = "979-8-89176-189-6",
    abstract = "Large Vision-Language Models (VLMs) have demonstrated strong capabilities in tasks requiring a fine-grained understanding of literal meaning in images and text, such as visual question-answering or visual entailment. However, there has been little exploration of the capabilities of these models when presented with images and captions containing figurative meaning, such as metaphors or humor. To close this gap, we propose a new task framing the figurative meaning understanding problem as an explainable visual entailment task, where the model has to predict whether the image (premise) entails a caption (hypothesis) and justify the predicted label with a textual explanation. The figurative phenomena can be present in the image, in the caption, or both. Using a human-AI collaboration approach, we build the accompanying expert-verified dataset V-FLUTE, containing 6,027 image, caption, label, explanation instances spanning five diverse figurative phenomena: metaphors, similes, idioms, sarcasm, and humor. Through automatic evaluation, we find that VLMs struggle to generalize from literal to figurative meaning, particularly when it is present in images. Further, we identify common types of errors in VLM reasoning (hallucination and incomplete or unsound reasoning) across classes of models via human evaluation."
}
```

# Contact

For any questions or additional information, please raise a github issue or contact [Arkadiy Saakyan](mailto:a.saakyan@cs.columbia.edu).

# Acknowledgment

This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200005. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

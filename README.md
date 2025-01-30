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
@misc{saakyan2024understanding,
    title={Understanding Figurative Meaning through Explainable Visual Entailment},
    author={Arkadiy Saakyan and Shreyas Kulkarni and Tuhin Chakrabarty and Smaranda Muresan},
    year={2024},
    eprint={2405.01474},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

# Contact

For any questions or additional information, please raise a github issue or contact [Arkadiy Saakyan](mailto:a.saakyan@cs.columbia.edu).

# Acknowledgment

This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200005. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

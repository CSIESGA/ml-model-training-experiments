# Machine Learning Training & Alignment Experiments

A collection of hands-on experiments exploring **Transformer sequence modeling**, **parameter-efficient supervised fine-tuning**, and **preference-based language model alignment**.

The notebooks document model configuration, hyperparameter adjustments, training runs, and evaluation of saved outputs. They use existing model architectures and starter training pipelines; my primary contributions were **configuring and tuning experiments, running training/inference, and analyzing recorded results**, rather than designing the underlying algorithms from scratch.

## Experiments

| Notebook | Focus | Recorded run |
| --- | --- | --- |
| [Pokémon Transformer Sequence Modeling](pokemon-transformer-sequence-modeling.ipynb) | Train a small autoregressive Transformer on Pokémon pixel sequences and inspect reconstruction/generation behavior | **100 epochs**; final logged training loss **0.0618**; validation reconstruction accuracy **37.23%** at epoch 100 |
| [Llama 2 LoRA Supervised Fine-Tuning](llama2-lora-supervised-finetuning.ipynb) | Configure parameter-efficient SFT of a quantized Llama 2 model and inspect inference outputs | **100 selected examples**, **2 configured epochs**, **24 logged training steps**; saved inference outputs |
| [Llama 3 DPO Preference Alignment](llama3-dpo-preference-alignment.ipynb) | Apply Direct Preference Optimization with LoRA and compare sample responses before and after training | **50 examples**, **3 configured epochs**, **18 completed steps**; final logged training loss **0.2247** |

> **Results note:** All numbers refer to specific outputs saved in the original notebooks, not independently rerun benchmarks. Training loss and selected examples are not evidence of generalization or a measured improvement over a baseline.

## Technical Highlights

### 1. Pokémon Transformer Sequence Modeling

- Configured and trained a six-layer Llama-style causal language model to model Pokémon image data as pixel sequences.
- Inspected training loss, validation reconstruction accuracy, and saved image previews over the training run.
- Explored the behavior of autoregressive sequence modeling on an image-oriented task.

**Stack:** PyTorch, Transformers (Llama configuration), Google Colab.

### 2. Llama 2 LoRA Supervised Fine-Tuning

- Configured a 4-bit Llama 2 7B training workflow with LoRA adapters and supervised fine-tuning.
- Adjusted training settings and data selection within an existing pipeline.
- Examined logged training progress and generated responses to evaluation inputs.

**Stack:** Python, Unsloth, Hugging Face Transformers, TRL, PEFT/LoRA, Google Colab.

### 3. Llama 3 DPO Preference Alignment

- Configured LoRA-based Direct Preference Optimization for a Llama 3 8B Instruct model using preference pairs.
- Tracked training progress and reviewed example model responses before and after alignment.
- Considered the difference between observed behavior on sample prompts and validated improvement on a held-out evaluation set.

**Stack:** Python, Unsloth, Hugging Face Transformers, TRL (DPOTrainer), PEFT/LoRA, Google Colab.

## Repository Structure

```text
ml-model-training-experiments/
├── README.md
├── pokemon-transformer-sequence-modeling.ipynb
├── llama2-lora-supervised-finetuning.ipynb
└── llama3-dpo-preference-alignment.ipynb
```

## Running the Notebooks

1. Open an `.ipynb` file in Google Colab.
2. Select an appropriate GPU runtime and follow the notebook's dependency setup cells.
3. Ensure external datasets and Hugging Face model permissions are available, where required.
4. Review the configuration and run cells sequentially. Compare your results with the saved outputs while accounting for GPU, software-version, dataset, and random-seed differences.

**Reproducibility:** The notebooks retain outputs from earlier executions, but their complete workflows have **not been freshly rerun or independently verified** for this repository. External model access, package compatibility, and computing resources may affect reproducibility.

## Scope and Provenance

The experiments adapt provided training scaffolds and established open-source model implementations. The original architecture designs and underlying methods (Transformer, LoRA, SFT, DPO) are **not claimed as original inventions**. Upstream credits and existing license notices should be preserved in the notebooks.

Before publishing, verify that all included starter code and datasets may be shared. Remove embedded credentials, private paths, sensitive identifiers, restricted data, and unnecessary large model artifacts. Do not assign a permissive license to third-party code unless its terms permit doing so.

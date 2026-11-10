# Machine Learning Training & Alignment Experiments

A collection of hands-on experiments exploring **Transformer sequence modeling**, **parameter-efficient supervised fine-tuning**, and **preference-based language model alignment**.

The notebooks document model configuration, hyperparameter adjustments, training runs, and evaluation of saved outputs. They use existing model architectures and starter training pipelines; my primary contributions were **configuring and tuning experiments, running training/inference, and analyzing recorded results**, rather than designing the underlying algorithms from scratch.

## Experiments

| Notebook | Focus | Recorded Run |
|---|---|---|
| [Pokémon Transformer Sequence Modeling](notebooks/pokemon-transformer-sequence-modeling.ipynb) | Autoregressive Transformer trained on Pokémon pixel sequences | 100 epochs; final training loss 0.0618; validation reconstruction accuracy 37.23% |
| [Llama 2 LoRA Supervised Fine-Tuning](notebooks/llama2-lora-supervised-finetuning.ipynb) | Parameter-efficient fine-tuning of a quantized Llama 2 model | 100 selected examples; 2 epochs; 24 training steps; inference outputs |
| [Llama 3 DPO Preference Alignment](notebooks/llama3-dpo-preference-alignment.ipynb) | Preference-based language model alignment | 50 examples; 3 configured epochs; 18 completed steps; training loss 0.2247 |

All reported results refer to specific saved experiment runs, not independently rerun benchmarks or measured improvements against baseline models.

## Technical Highlights

### 1. Pokémon Transformer Sequence Modeling

- Configured and trained a six-layer Llama-style causal language model for Pokémon pixel-sequence modeling.
- Analyzed training loss, validation reconstruction accuracy, and generated image previews.
- Explored autoregressive sequence modeling on an image-oriented task.

**Technologies:** PyTorch, Hugging Face Transformers, Google Colab.

### 2. Llama 2 LoRA Supervised Fine-Tuning

- Configured a 4-bit Llama 2 7B training pipeline with LoRA adapters.
- Adjusted training hyperparameters and data-selection settings.
- Examined training progress and generated model responses.

**Technologies:** Python, Unsloth, Transformers, TRL, PEFT/LoRA.

### 3. Llama 3 DPO Preference Alignment

- Configured Direct Preference Optimization for a Llama 3 8B Instruct model.
- Applied LoRA-based fine-tuning with preference-pair data.
- Compared sample model responses before and after alignment.

**Technologies:** Python, Unsloth, Transformers, TRL, PEFT/LoRA.

## Repository Structure

```text
ml-model-training-experiments/
├── README.md
├── pokemon-transformer-sequence-modeling.ipynb
├── llama2-lora-supervised-finetuning.ipynb
└── llama3-dpo-preference-alignment.ipynb
```

## Running the Experiments

1. Open the selected notebook in Google Colab.
2. Select an appropriate GPU runtime.
3. Install dependencies using the notebook's setup cells.
4. Ensure any required datasets and model permissions are available.
5. Execute the notebook sequentially and inspect the saved training and evaluation outputs.

Results may vary with software versions, random seeds, GPU availability, and dataset access. End-to-end reproducibility has not been independently reverified.

## Scope and Attribution

These experiments build on existing model architectures, open-source libraries, and provided training scaffolds.

My contributions primarily involved experiment configuration, parameter tuning, execution, and analysis of recorded results. The Transformer, LoRA, SFT, and DPO algorithms are not claimed as original inventions.

Original attribution and license notices should be preserved. Before public release, verify redistribution permissions and remove credentials, private paths, restricted assets, and sensitive information.

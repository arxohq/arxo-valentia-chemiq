# Arxo Valentia ChemIQ 0.1: a 4B model reaches 74.1% on ChemIQ

Arxo Valentia ChemIQ 0.1 answers organic chemistry questions from a SMILES string without tools: it counts atoms and rings, finds the shortest path between two atoms, maps the atoms of two SMILES strings of one molecule, writes an IUPAC name, predicts a score from a structure-activity table and writes the product of a reaction. This is `Arxo-Valentia-4B-ChemIQ-0.1`, a 4B model based on [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The model is on Hugging Face: [arxohq/Arxo-Valentia-4B-ChemIQ-0.1](https://huggingface.co/arxohq/Arxo-Valentia-4B-ChemIQ-0.1).

It follows the Arxo Materia releases on [MedCalc-Bench](https://github.com/arxohq/arxo-medcalc-bench) and [EHRSQL](https://github.com/arxohq/arxo-materia-ehrsql).

## Results

[ChemIQ](https://github.com/oxpig/ChemIQ), all 816 questions, scored by the benchmark's own parser and answer checker. The model answers without tools.

| System | Trained on the task | ChemIQ | Source |
|---|---|---|---|
| **Arxo Valentia ChemIQ 0.1 (4B)** | yes | **605 / 816 (74.1%)** | this model |
| GPT-5 | no | 70.2% | [post by the benchmark's author](https://www.blopig.com/blog/2025/08/gpt-5-achieves-state-of-the-art-chemical-intelligence/) |
| o3-mini, high reasoning | no | 56.7% | [ChemIQ paper](https://arxiv.org/abs/2505.07735) |
| Gemini 2.5 Pro | no | 55.5% | ChemIQ paper |
| DeepSeek-R1 | no | 50.2% | ChemIQ paper |

One training run. Greedy decoding, thinking switched off, a limit of 3,900 new tokens. Arxo Valentia was trained on questions of the same kinds, built from molecules outside the test set; the other rows are general models that answer from the prompt alone.

## Results by category

| Category | Questions | Arxo Valentia ChemIQ 0.1 (4B) | o3-mini, high | Gemini 2.5 Pro | DeepSeek-R1 |
|---|---|---|---|---|---|
| Carbon counting | 50 | **50 (100%)** | 92.0% | 82.0% | 90.0% |
| Ring counting | 48 | **48 (100%)** | 100% | 93.8% | 93.8% |
| Shortest path | 108 | **108 (100%)** | 72.2% | 63.0% | 59.3% |
| Atom mapping | 184 | **144 (78.3%)** | 56.0% | 46.2% | 54.3% |
| SMILES to IUPAC name | 200 | **141 (70.5%)** | 28.5% | 38.5% | 20.0% |
| Structure-activity (Free-Wilson) | 40 | 34 (85.0%) | 100% | 95.0% | 95.0% |
| Reaction product | 90 | **80 (88.9%)** | 60.0% | 55.6% | 45.6% |
| NMR structure elucidation | 96 | 0 (0%) | 38.5% | 51.0% | 38.5% |
| **All** | **816** | **605 (74.1%)** | 56.7% | 55.5% | 50.2% |

The columns of the other models are computed from the model responses released with the benchmark, with the same checker; their totals match the paper.

Two notes on the numbers:

- The model was not trained on NMR questions and solves none of them. On the other seven categories it scores 605 of 720 (84.0%).
- The benchmark's checker rejects any answer longer than 200 characters. An atom mapping for a molecule of 23 or more atoms is longer than that, so 30 of the 184 atom-mapping questions cannot be credited to any answer. With that limit lifted the same answers score 629 of 816 (77.1%), and atom mapping 168 of 184.

## How to use

The adapter attaches to `AutoModelForImageTextToText`, not to `AutoModelForCausalLM`. The system prompt below is the one the model was trained with; the question follows the wording of ChemIQ.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "arxohq/Arxo-Valentia-4B-ChemIQ-0.1")

SYSTEM = (
    "You are a careful chemist. Work in the fixed step-by-step notation: write out the working first, "
    "each step using only what is already written, then give the final answer alone on the last line "
    "in exactly the form the question asks for."
)
question = (
    "Write the IUPAC name of this molecule:\n\n"
    "CC(=O)Nc1ccc(O)cc1\n\n"
    "Write the IUPAC name only. Do not write any comments."
)

messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=3900, do_sample=False)
answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(answer.strip().splitlines()[-1])
```

The model writes its working and then the answer alone on the last line. The benchmark's parser reads that last line.

## Scope

- The model covers seven of the eight ChemIQ categories and the question wording of the benchmark. NMR structure elucidation is outside this release. On other chemistry questions it is untested.
- A name counts as correct when [OPSIN](https://github.com/dan2097/opsin) parses it to the intended structure, as ChemIQ scores it. The model writes a systematic name of that kind; it is not always the preferred IUPAC name.
- Molecules are those of the benchmark's range: neutral organic molecules of up to about 30 heavy atoms, without stereochemistry.
- It is a research release.

## Files

| Path | What |
|---|---|
| `answers/4b.jsonl` | the model's output for each of the 816 questions, by question id |
| `eval/4b.scores.jsonl` | per-question result from the benchmark's checker, with and without the 200-character limit |
| `adapters/4b.adapter_config.json` | adapter configuration |
| release `v0.1`, `valentia-4b-chemiq.safetensors` | adapter weights; save as `adapter_model.safetensors` next to the configuration |
| `SHA256SUMS` | checksums of all of the above |

To check the scores, run the parser and the answer checker from [oxpig/ChemIQ](https://github.com/oxpig/ChemIQ) on `answers/4b.jsonl`; names are resolved with OPSIN.

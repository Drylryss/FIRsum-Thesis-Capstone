"""Offline dependency smoke check. Does not download models or train thesis data."""
import importlib
import tempfile
import core  # Configure the project-local Hugging Face cache before ML imports.
from pathlib import Path


def main():
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'datasets', 'numpy',
                 'sklearn', 'sentencepiece', 'fastapi', 'uvicorn', 'pypdf',
                 'rouge_score', 'sacrebleu', 'bert_score', 'httpx', 'multipart', 'google.protobuf'):
        module = importlib.import_module(name)
        print(f'{name}: {getattr(module, "__version__", "import OK")}')
    import torch
    from transformers import BertConfig, BertForSequenceClassification, PegasusConfig, PegasusForConditionalGeneration
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    config = BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64, num_labels=4)
    model = get_peft_model(BertForSequenceClassification(config),
        LoraConfig(task_type=TaskType.SEQ_CLS, r=2, lora_alpha=4,
                   target_modules=['query', 'value'], modules_to_save=['classifier']))
    inputs = {'input_ids': torch.tensor([[2, 3, 4]]), 'attention_mask': torch.ones(1, 3, dtype=torch.long)}
    model(**inputs, labels=torch.tensor([0])).loss.backward()
    temp_root = Path(__file__).resolve().parent / '.cache' / 'checks'
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='firmsum-check-', dir=temp_root) as folder:
        model.save_pretrained(folder)
        restored = PeftModel.from_pretrained(BertForSequenceClassification(config), folder)
        assert restored(**inputs).logits.shape == (1, 4)
    tiny = PegasusForConditionalGeneration(PegasusConfig(vocab_size=100, d_model=32,
        encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
        decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64,
        max_position_embeddings=64, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0))
    tiny.eval()
    with torch.inference_mode():
        assert tiny.generate(**inputs, max_new_tokens=4).shape[0] == 1
    print(f'CPU execution OK; CUDA available: {torch.cuda.is_available()}')
    print('Tiny random BERT LoRA backward/save/reload and Pegasus generation passed.')
    print('This checks the software installation, not trained model accuracy.')


if __name__ == '__main__':
    main()

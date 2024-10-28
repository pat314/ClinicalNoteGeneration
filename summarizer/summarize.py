from transformers import BartForConditionalGeneration, BartTokenizer, PreTrainedTokenizer

def summarize_text(text: str,
                    model: Union[PreTrainedModel, str] = "facebook/bart-large-cnn",
                    tokenizer: Union[PreTrainedTokenizer, str] = None,
                    min_length: int = 30,
                    max_length: int = 130,
                    device: str = 'cpu') -> str:
    if isinstance(model, str):
        model_name = model
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
    model = model.to(device)

    encoder_inputs = tokenizer(text, truncation = True, max_length=1024, return_tensors='pt')
    inputs = encoder_inputs
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    summary_ids = model.generate(**inputs, min_length=min_length, max_length=max_length)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

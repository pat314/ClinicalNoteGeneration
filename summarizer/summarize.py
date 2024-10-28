from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Union

def summarize_text(
        text: str,
        model: Union[PreTrainedModel, str] = "facebook/bart-large-cnn",
        tokenizer: Union[PreTrainedTokenizer, str] = None,
        min_length: int = 30,
        max_length: int = 130,
        device: str = 'cpu'
) -> str:
    """
    Summarizes the given text using a BART model.

    Args:
        text (str): The text to be summarized.
        model (Union[PreTrainedModel, str]): The pre-trained BART model or its name. Default is "facebook/bart-large-cnn".
        tokenizer (Union[PreTrainedTokenizer, str], optional): The tokenizer to use with the model. 
        min_length (int): Minimum length of the summary. Default is 30.
        max_length (int): Maximum length of the summary. Default is 130.
        device (str): Device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        str: The generated summary of the input text.
    """
    # Load model and tokenizer if a string is provided
    if isinstance(model, str):
        model_name = model
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
    else:
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided if model is not a string.")

    # Move model to the specified device
    model = model.to(device)


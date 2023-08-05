import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
import torch

# Constants
SAMPLERATE = 16000  # Hz
DURATION = 3  # seconds to record

MAX_PAST_INTERACTIONS = 10  # Maximum interactions to remember
TEMPERATURE = 0.9
TOP_K = 150
TOP_P = 1.0
REPETITION_PENALTY = 1.2
BOT_NAME = "Siri"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def initialize_speech_model(model_name='openai/whisper-medium'):
    """Initialize and return the speech recognition model."""
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    return processor, model


def initialize_language_model(model_name='gpt2-xl'):
    """Initialize and return the language model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    return tokenizer, model


def record_and_transcribe(processor, speach_model):
    """Record user's speech and transcribe it."""
    input("Press Enter to speak...")

    myrecording = sd.rec(int(SAMPLERATE * DURATION), samplerate=SAMPLERATE, channels=1)
    print("Recording...")
    sd.wait()
    print("Recording finished")

    input_features = processor(myrecording[:, 0].tolist(), sampling_rate=SAMPLERATE, return_tensors="pt").input_features
    predicted_ids = speach_model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print(transcription)
    return transcription


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (nucleus filtering)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Create mask
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[mask] = filter_value

    return logits


def interact_with_model(speech_processor, speech_model, tokenizer, language_model, user_name, temperature, top_k, top_p, repetition_penalty):

    """Interact with the user using speech recognition and the GPT-2 model."""
    # To store past interactions
    past_interactions = []
    generated_sequences = set()  # Keep track of generated sequences to avoid repetition

    while True:  # infinity prompt
        speach_input = record_and_transcribe(speech_processor, speech_model)
        if not speach_input.strip():
            print("Couldn't detect speech. Try again.")
            continue

        # Add the recent interaction to the list and truncate if exceeds limit
        past_interactions.append(f"-{user_name}: {speach_input}.")
        past_interactions = past_interactions[-MAX_PAST_INTERACTIONS:]

        # Construct context
        context = "\n".join(past_interactions) + " -" + BOT_NAME + ": "


        encoding = tokenizer.encode_plus(context, return_tensors='pt', add_special_tokens=False, truncation=True, max_length=1024)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        generated_tokens_count = 0
        while True:
            with torch.no_grad():
                outputs = language_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            # Apply repetition penalty
            for token_seq in generated_sequences:
                for token in token_seq:
                    logits[0, -1, token] /= repetition_penalty

            logits = logits[:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)

            generated_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(generated_text, end='', flush=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)

            generated_tokens_count += 1
            
            # Check for punctuation or token count limit to end the generation
            if generated_text in ['.', '!', '?', user_name] or generated_tokens_count >= 100:  # Adjust the 100 as needed
              print("\n")
              generated_sequence = input_ids[:, -generated_tokens_count:].squeeze().tolist()
              if isinstance(generated_sequence, list) and all(isinstance(i, int) for i in generated_sequence):
                generated_sequences.add(tuple(generated_sequence))
              else:
                print(f"Unexpected type or content: {generated_sequence}")
              past_interactions.append("-Siri: " + tokenizer.decode(generated_sequence))
              break


if __name__ == "__main__":
    speech_processor, speech_model = initialize_speech_model()
    tokenizer, language_model = initialize_language_model()
    user_name = input("What is your name? ")
    interact_with_model(speech_processor, speech_model, tokenizer, language_model, user_name, TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY)

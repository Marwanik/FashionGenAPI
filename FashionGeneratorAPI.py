from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from diffusers import StableDiffusionPipeline
import base64
from io import BytesIO

app = Flask(__name__)

# Model 1 - Translation
translation_checkpoint = "Helsinki-NLP/opus-mt-ar-en"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_checkpoint)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_checkpoint)
translation_model = translation_model.to("cuda")

LANG_TOKEN_MAPPING = {
    'en': '<en>',
    'ar': '<ar>',
}

def encode_input_str(text, target_lang, tokenizer, seq_len, lang_token_map=LANG_TOKEN_MAPPING):
    target_lang_token = lang_token_map[target_lang]
    input_ids = tokenizer.encode(
        text=target_lang_token + text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=seq_len)
    return input_ids[0].to("cuda")

def translate_text(input_text, target_lang):
    input_ids = encode_input_str(
        text=input_text,
        target_lang=target_lang,
        tokenizer=translation_tokenizer,
        seq_len=translation_model.config.max_length,
        lang_token_map=LANG_TOKEN_MAPPING)
    input_ids = input_ids.unsqueeze(0)
    output_tokens = translation_model.generate(input_ids, num_beams=20, length_penalty=0.2)
    output_text = translation_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text

# Model 2 - Image Generation
image_pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float32)
image_pipe.to("cuda")

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    input_text = data['input_text']
    target_lang = 'en'  # Set target_lang to 'en' (English) by default

    output_text = translate_text(input_text, target_lang)

    print(f"Generating an image of: {output_text}")
    image = image_pipe(output_text).images[0]
    print("Image generated! Converting image...")

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    b = "data:image/png;base64," + str(img_str)[2:-1]

    response = {
        'generated_image': b
    }
    print("Sending image...")
    return jsonify(response)

if __name__ == '__main__':
    app.run()

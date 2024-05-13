import tempfile
from flask import request, render_template
from flask import Flask

app = Flask(__name__)
from transformers import BertTokenizerFast, EncoderDecoderModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization')
model = EncoderDecoderModel.from_pretrained('mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization').to(device)


def generate_summary(text):
    # cut off at BERT max length 512
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    output = model.generate(input_ids, attention_mask=attention_mask)

    return tokenizer.decode(output[0], skip_special_tokens=True)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize',methods=['POST'])
def predict():  # put application's code here
    text = request.form.get('text')

    return render_template('index.html', text=text, summary=generate_summary(text))
if __name__ == '__main__':
    app.run()

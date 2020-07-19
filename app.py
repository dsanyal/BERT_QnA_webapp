import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
from flask import Flask,render_template,url_for,request

app = Flask(__name__, template_folder='templates')

#memory = joblib.Memory("models/", verbose=0)

def inference(question, context):
  input_dict = tokenizer.encode_plus(question, context, return_tensors='tf')
  # print(input_dict)
  start_scores, end_scores = model(input_dict)
  all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
  answer = ' '.join(all_tokens[tf.math.argmax(start_scores, 1)[0] : tf.math.argmax(end_scores, 1)[0]+1])
  return answer

@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
      return render_template('index.html')
    if request.method == 'POST':
      question, context = request.form['question'], request.form['context']
      answer = inference(question, context)
      return render_template('index.html', result = answer, question=question, context=context)



if __name__ == '__main__':
  tokenizer = DistilBertTokenizer.from_pretrained("./models/tokenizer")
  model = TFDistilBertForQuestionAnswering.from_pretrained("./models")
  app.run(debug=True)

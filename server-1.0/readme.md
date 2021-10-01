# Taken from ua-gec
# I have removed some punctuation to better see the effects of the model
test_text = """Де мені краще жити в квартирі чи у приватному будинку?

Таке питання я задаю собі постійно, і час від часу відповідаю собі різносторонньо.

Квартира дає такі переваги як мобільність та доступність на противагу заміського будинку де навіть  щоб купити продукти потрібно буде послуговуватися транспортним засобом. Хоча разом з тим, віддаленість будинку краще для здоровя людини, як фізичного так і психологічного.

Заміський будинок може бути копією раю на цій землі Підстрижена зелена травичка хвойні дерева невеличкий басейн, кущики корисних ягід для карапузиків, романтичні вечірні посиденьки під зоряним небом - це те, що може зсолоджувати наше житя закарбовувати незабутні миті!

Сусіди. Напевно це саме болюче питання. Квартира передбачає багато сусідів а багато сусідів - це багато подразників. Можна незважати на більшість з них але ж куди дітися від кухонного штину тьоті Зіни або затхлого запаху черевиків дяді Толі?
В приватному будинку хоч і менше сусідів але буває що деякі можуть замінити здесяток квартирних. Чого вартує тьотя Люба з її величезною кучею відходів домашнього тваринництва.

Та все ж таки розмірковуючи роками про вибір де проживати в квартирі чи будинку я приходжу до висновку, що молодим жаданніше жити в міській квартирі, а вже доросліючи  - перебиратися в заміський будинок.
"""


# To split into sentences
import nltk
nltk.download('punkt')

import re

# Removes extra spaces before punctuation
def remove_extra_spaces(sent):
  return re.sub(r'\s+([?,:;–.!"])', r'\1', sent)

# Adds extra spaces before punctuation
def add_extra_spaces(sent):
  sent = re.sub('([.,!?()])', r' \1 ', sent)
  sent = re.sub('\s{2,}', ' ', sent)
  return sent

# Add extra spaces before sending to punct model
test_text = add_extra_spaces(test_text)
test_sent = nltk.sent_tokenize(test_text)


from flask import Flask, request, jsonify
from model import predict_sentence

application = Flask(__name__)
@application.route('/')
def hello():
    print("Hello World!")
    return "<h1>Hello World!</h1>"

@application.route('/predict', methods=["GET", "POST"])
def predict():
    input_data = request.get_json(force=True)
    text = input_data.get('input')
    output = {'result':predict_sentence(text)}
    return jsonify(output)

if __name__ == "__main__":
    application.run(host='0.0.0.0')

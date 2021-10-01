import os
dirname = os.path.dirname(__file__)
MODEL_FOLDER = os.path.join(dirname, 'models/lenguist')

from tagpredictor import TagPredictor
pred = TagPredictor()

pred.load_model(model_path=MODEL_FOLDER + '/best.th',
                vocab_path=MODEL_FOLDER + '/vocabulary')

test_sent = ["Привіт світ як справи в мене норм", "Житття хуйня"]
out = pred.predict(test_sent)
print(out)

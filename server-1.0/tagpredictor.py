import torch
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.data import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.nn import util

from model import Seq2Labels

class TagPredictor(object):
  """
  Input: List of sentences
  Output: List of dictionaries {sentence:"", edits:[]}

  Initialization:
  - batch_size (def = 64) batch size to use when processing. Should be as high as memory allows.
  - additional_confidence (def = 0.1) we add this value to the model's top prediction probability before running any other checks.
    Generally, the idea is to regulate somewhat the impact of min_error_check
  - min_error_probability (def = 0.3) minimum threshold of confidence in an error model needs to have in order to try and correct anything.
    If probability of non-KEEP tag is less than min_error_probability, we do not apply it. Helps curb the number of False Positives.

  """
  def __init__(self,
                batch_size = 64,
                additional_confidence = 0.1,
                min_error_probability = 0.4,
                log = False
                ):
    self.batch_size = batch_size
    self.additional_confidence = additional_confidence
    self.min_error_probability = min_error_probability

  # Loads model from files
  def load_model(self, model_path, vocab_path):
    transformer_name = "youscan/ukr-roberta-base"
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Vocabulary
    self.vocab = Vocabulary.from_files(vocab_path)
    # Indexer
    bert_token_indexer = PretrainedTransformerMismatchedIndexer(model_name=transformer_name)
    self.indexer = {'indexer': bert_token_indexer}
    # Embedder
    token_embedder = PretrainedTransformerMismatchedEmbedder(model_name=transformer_name,
                                                            train_parameters=False,
                                                            last_layer_only=True)
    text_field_embedder = BasicTextFieldEmbedder({'indexer':token_embedder})
    # Model
    model = Seq2Labels(vocab=self.vocab,
                      text_field_embedder=text_field_embedder,
                      ).to(self.device)
    if torch.cuda.is_available():
      model.load_state_dict(torch.load(model_path))
    else:
      model.load_state_dict(torch.load(model_path,
                            map_location=torch.device('cpu')))
    model.eval()
    self.model = model

  # Splits one list of sentences into list of batches of sentences
  def split_into_batches(self, sentences):
    batches = []
    batch = []
    for sentence in sentences:
      batch.append(sentence.split())
      if len(batch) == self.batch_size:
        batches.append(batch)
        batch = []
    if batch:
      batches.append(batch)
    return batches

  # Converts batches into Allen NLP instances
  def preprocess_batch(self, sent_batch):
    token_batch = []
    for sentence in sent_batch:
      tokens = sentence
      # We add start token to our full token list
      tokens = [Token(token) for token in ['$START'] + sentence]
      token_batch.append(Instance({'tokens': TextField(tokens, self.indexer)}))
    batch = Batch(token_batch)
    batch.index_instances(self.vocab)
    return batch

  # Applies model to a batch
  def run_through_model(self, batch):
    batch = util.move_to_device(batch.as_tensor_dict(), 0 if torch.cuda.is_available() else -1)
    with torch.no_grad():
        model_output = self.model.forward(**batch)
    max_vals = torch.max(model_output['class_probabilities_labels'], dim=-1)
    # Probabilities of top labels
    probs = max_vals[0].tolist()
    # Indexes in vocab of the top labels
    idx = max_vals[1].tolist()
    return probs, idx

  # Applies additional checks to boost performance
  def apply_checks(self, batch_probs, batch_idx):
    for probs, idx in zip(batch_probs, batch_idx):
      # Adds additional confidence
      for prob in probs:
        prob = prob + self.additional_confidence

      #Apply min_error_prob check
      for i in range(len(probs)):
        # If prob is not high enough, make tag be KEEP
        if probs[i] < self.min_error_probability:
          idx[i] = 0
    return batch_probs, batch_idx

  # Final predict function
  def predict(self, sentences):
    # Split list of sentences into list of batches of sentences
    batches = self.split_into_batches(sentences)
    prediction = []
    for batch in batches:
      # Preprocess every batch
      preprocessed_batch = self.preprocess_batch(batch)
      # Run predictions for every batch
      probs, idx = self.run_through_model(preprocessed_batch)
      # Run checks for every batch prediction output
      probs, idx = self.apply_checks(probs, idx)
      # Append words and tags to output object
      for sentence, tags in zip(batch, idx):
        sentence = ['$START'] + sentence
        output = {'tokens':[], 'tags':[]}
        for i in range(len(sentence)):
          output['tokens'].append(sentence[i])
          tag = self.vocab.get_token_from_index(tags[i],namespace='labels')
          output['tags'].append(tag)
        prediction.append(output)
    return prediction

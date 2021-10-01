#Define model
from typing import Dict, Optional, List, Any

import numpy
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from torch.nn.modules.linear import Linear

class Seq2Labels(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 labels_namespace: str = "labels",
                 detect_namespace: str = "d_tags",
                 predictor_dropout=0.0,
                 label_smoothing: float = 0.0,
                 confidence: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 **kwargs) -> None:
        super(Seq2Labels, self).__init__(vocab, **kwargs)

        self.label_namespaces = [labels_namespace,
                                 detect_namespace]
        self.text_field_embedder = text_field_embedder
        self.encoder = PassThroughEncoder(input_dim=self.text_field_embedder.get_output_dim())
        self.num_labels_classes = self.vocab.get_vocab_size(labels_namespace)
        self.num_detect_classes = self.vocab.get_vocab_size(detect_namespace)
        self.label_smoothing = label_smoothing
        self.confidence = confidence
        self.incorr_index = self.vocab.get_token_index("INCORRECT",
                                                       namespace=detect_namespace)

        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(predictor_dropout))

        self.tag_labels_projection_layer = TimeDistributed(
            Linear(self.encoder.get_output_dim(), self.num_labels_classes))

        self.tag_detect_projection_layer = TimeDistributed(
            Linear(self.encoder.get_output_dim(), self.num_detect_classes))

        self.metrics = {"accuracy": CategoricalAccuracy()}

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                d_tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)

        logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(encoded_text))
        logits_d = self.tag_detect_projection_layer(encoded_text)

        class_probabilities_labels = F.softmax(logits_labels, dim=-1).view(
            [batch_size, sequence_length, self.num_labels_classes])
        class_probabilities_d = F.softmax(logits_d, dim=-1).view(
            [batch_size, sequence_length, self.num_detect_classes])
        error_probs = class_probabilities_d[:, :, self.incorr_index] * mask
        incorr_prob = torch.max(error_probs, dim=-1)[0]

        if self.confidence > 0:
            probability_change = [self.confidence] + [0] * (self.num_labels_classes - 1)
            class_probabilities_labels += torch.FloatTensor(probability_change).repeat(
                (batch_size, sequence_length, 1)).to(class_probabilities_labels.device)

        output_dict = {"logits_labels": logits_labels,
                       "logits_d_tags": logits_d,
                       "class_probabilities_labels": class_probabilities_labels,
                       "class_probabilities_d_tags": class_probabilities_d,
                       "max_error_probability": incorr_prob}
        if labels is not None and d_tags is not None:
            loss_labels = sequence_cross_entropy_with_logits(logits_labels, labels, mask,
                                                             label_smoothing=self.label_smoothing)
            loss_d = sequence_cross_entropy_with_logits(logits_d, d_tags, mask)
            for metric in self.metrics.values():
                metric(logits_labels, labels, mask.float())
                metric(logits_d, d_tags, mask.float())
            output_dict["loss"] = loss_labels + loss_d

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        for label_namespace in self.label_namespaces:
            all_predictions = output_dict[f'class_probabilities_{label_namespace}']
            all_predictions = all_predictions.cpu().data.numpy()
            if all_predictions.ndim == 3:
                predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
            else:
                predictions_list = [all_predictions]
            all_tags = []

            for predictions in predictions_list:
                argmax_indices = numpy.argmax(predictions, axis=-1)
                tags = [self.vocab.get_token_from_index(x, namespace=label_namespace)
                        for x in argmax_indices]
                all_tags.append(tags)
            output_dict[f'{label_namespace}'] = all_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}
        return metrics_to_return

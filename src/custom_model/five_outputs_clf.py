from typing import Dict, Optional

import numpy as np
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics.fbeta_measure import FBetaMeasure

from src.product_dataset import ProductDataset


@Model.register('five_levels_clf', exist_ok=True)
class FiveOutputsTextClassifier(Model):
    """
    This `Model` implements a weighted text classifier with five outputs. After embedding the text into
    a text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`. The
    resulting sequence is pooled using a `Seq2VecEncoder` and then passed to
    FIVE classification layers, which project into the label space. If a
    `Seq2SeqEncoder` is not provided, we will pass the embedded text directly to the
    `Seq2VecEncoder`.

    Registered as a `Model` with name "five_outputs".

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    seq2seq_encoder : `Seq2SeqEncoder`, optional (default=`None`)
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder : `Seq2VecEncoder`
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    feedforward : `FeedForward`, optional, (default = `None`)
        An optional feedforward layer to apply after the seq2vec_encoder.
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    num_labels1 : `int`, optional (default = `None`)
    num_labels2 : `int`, optional (default = `None`)
    num_labels3 : `int`, optional (default = `None`)
    num_labels4 : `int`, optional (default = `None`)
    num_labels5 : `int`, optional (default = `None`)
        Number of labels to project to in classification layers. By default, the classification layers will
        project to the size of the vocabulary namespace corresponding to labels.
    namespace : `str`, optional (default = `"tokens"`)
        Vocabulary namespace corresponding to the input text. By default, we use the "tokens" namespace.
    label_namespace1 : `str`, optional (default = `"labels1"`)
    label_namespace2 : `str`, optional (default = `"labels2"`)
    label_namespace3 : `str`, optional (default = `"labels3"`)
    label_namespace4 : `str`, optional (default = `"labels4"`)
    label_namespace5 : `str`, optional (default = `"labels5"`)
        Vocabulary namespace corresponding to labels of each level. By default, we use the "labels1 ... labels5" namespace.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    class_weights : 'list', deprecated; actual class weights for outputs 1..5 are detalized in the __init__()
    """

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            seq2vec_encoder: Seq2VecEncoder,
            seq2seq_encoder: Seq2SeqEncoder = None,
            feedforward: Optional[FeedForward] = None,
            dropout: float = None,
            num_labels1: int = None,
            num_labels2: int = None,
            num_labels3: int = None,
            num_labels4: int = None,
            num_labels5: int = None,
            label_namespace1: str = "labels1",
            label_namespace2: str = "labels2",
            label_namespace3: str = "labels3",
            label_namespace4: str = "labels4",
            label_namespace5: str = "labels5",
            namespace: str = "tokens",
            data_file: str = None,
            foldnum: int = None,
            with_hierarchical_loss: bool = True,
            with_connected_outputs: bool = True,
            loss_alpha: float = 1,
            loss_beta: float = 0.8,
            initializer: InitializerApplicator = InitializerApplicator(),
            device: int = -1,
            **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._label_namespace1 = label_namespace1
        self._label_namespace2 = label_namespace2
        self._label_namespace3 = label_namespace3
        self._label_namespace4 = label_namespace4
        self._label_namespace5 = label_namespace5
        self._namespace = namespace

        if num_labels1:
            self._num_labels1 = num_labels1
            self._num_labels2 = num_labels2
            self._num_labels3 = num_labels3
            self._num_labels4 = num_labels4
            self._num_labels5 = num_labels5
        else:
            raise BaseException('Specify number of labels for levels 1, 2, 3, 4, 5')

        self._with_connected_outputs = with_connected_outputs
        input1_dim = self._classifier_input_dim
        input2_dim = self._classifier_input_dim + self._num_labels1 if self._with_connected_outputs else self._classifier_input_dim
        input3_dim = self._classifier_input_dim + self._num_labels2 if self._with_connected_outputs else self._classifier_input_dim
        input4_dim = self._classifier_input_dim + self._num_labels3 if self._with_connected_outputs else self._classifier_input_dim
        input5_dim = self._classifier_input_dim + self._num_labels4 if self._with_connected_outputs else self._classifier_input_dim

        self._classification_layer1 = torch.nn.Linear(input1_dim, self._num_labels1)
        self._classification_layer2 = torch.nn.Linear(input2_dim, self._num_labels2)
        self._classification_layer3 = torch.nn.Linear(input3_dim, self._num_labels3)
        self._classification_layer4 = torch.nn.Linear(input4_dim, self._num_labels4)
        self._classification_layer5 = torch.nn.Linear(input5_dim, self._num_labels5)

        self.metrics = {"f1_1": FBetaMeasure(average='macro'),
                        "f1_2": FBetaMeasure(average='macro'),
                        "f1_3": FBetaMeasure(average='macro'),
                        "f1_4": FBetaMeasure(average='macro'),
                        "f1_5": FBetaMeasure(average='macro'),
                        }

        device = 'cpu' if device == -1 else f'cuda:{device}'
        self._device = torch.device(device)

        dataset = ProductDataset.from_pickle(data_file)
        branches = dataset.branches
        train_df, _, _ = dataset.get_split(foldnum)
        self._init_loss(with_hierarchical_loss, loss_alpha, loss_beta, branches, train_df)

        initializer(self)

    def count_class_weights(self, data):
        res = []
        true_length = [self._num_labels1, self._num_labels2, self._num_labels3, self._num_labels4, self._num_labels5]
        for i, col in enumerate(sorted([c for c in data.columns if c.startswith('LVL_')])):
            if data[col].dtype == 'object':
                data[col] = data[col].fillna('nan').astype('category')  # for 100x faster value_counts
            vc = 1 - data[col].value_counts(normalize=True)
            current_weights = vc / vc.sum()
            weights = np.zeros(true_length[i], dtype=current_weights.dtype)  # e.g. len=172 while len(cw)=171
            weights[:len(current_weights)] = current_weights
            res.append(weights)

        return res

    def _init_loss(self, with_hierarchical_loss, alpha, beta, branches, train_df):
        def _init_lloss():
            """ Cross entropy losses for each output """

            lvl_weights = self.count_class_weights(train_df)
            class_weights = [
                torch.FloatTensor(lvl_i_weights).to(self._device) for lvl_i_weights in lvl_weights
            ]
            self._lloss = [torch.nn.CrossEntropyLoss(weight=class_weight) for class_weight in class_weights]

        def _init_hloss():
            """ Loss penalizing non-existent hierarchical dependencies. """

            def get_token_index(label, namespace, default_label='nan'):
                default_index = self.vocab._token_to_index[namespace].get(default_label, 0)
                return self.vocab._token_to_index[namespace].get(label, default_index)

            self._true_hierarchy_deps = [torch.zeros((self._num_labels1, self._num_labels2), dtype=torch.bool),
                                         torch.zeros((self._num_labels2, self._num_labels3), dtype=torch.bool),
                                         torch.zeros((self._num_labels3, self._num_labels4), dtype=torch.bool),
                                         torch.zeros((self._num_labels4, self._num_labels5), dtype=torch.bool)]
            for branch in branches:
                path = [get_token_index(label, f'labels{lvl + 1}') for lvl, label in enumerate(branch.split(' | '))]
                for i in range(1, len(path)):
                    self._true_hierarchy_deps[i - 1][path[i - 1], path[i]] = True

        _init_lloss()

        self._with_hierarchical_loss = with_hierarchical_loss
        if self._with_hierarchical_loss:
            self._loss_alpha = alpha
            self._loss_beta = beta
            _init_hloss()

    def loss(self, all_logits, all_labels):
        def _lloss():
            llosses = []
            for level, (lloss_f, logits, labels) in enumerate(zip(self._lloss, all_logits, all_labels)):
                if labels is not None:
                    llosses.append(lloss_f(logits.to(self._device), labels.long().to(self._device)))
                    self.metrics[f'f1_{level + 1}'](logits, labels)

            return sum(llosses) / 5.

        def _hloss():
            batch_size = all_logits[0].shape[0]
            predicted_paths = [[] for _ in range(batch_size)]
            probs = [torch.nn.functional.softmax(logits, dim=-1) for logits in all_logits]
            for lvl_proba in probs:
                pred_idx = lvl_proba.argmax(dim=-1)
                for batch in range(pred_idx.shape[0]):
                    predicted_paths[batch].append(pred_idx[batch])

            hlosses = []
            for path in predicted_paths:
                hloss = 0.
                for i in range(1, len(path)):
                    is_true_dependency = self._true_hierarchy_deps[i - 1][path[i - 1], path[i]]
                    hloss += self._loss_beta * (1 - float(is_true_dependency))

                hlosses.append(hloss / 5.)

            return sum(hlosses)

        # Weighted sum with hierarchical loss (optional)
        # Loss = alpha * CELoss + beta * HLoss, alpha >= 0, 0 <= beta <= 1
        if self._with_hierarchical_loss:
            return self._loss_alpha * _lloss() + _hloss()
        else:
            return _lloss()

    def forward(
            self,
            tokens: TextFieldTensors,
            label1: torch.IntTensor = None,
            label2: torch.IntTensor = None,
            label3: torch.IntTensor = None,
            label4: torch.IntTensor = None,
            label5: torch.IntTensor = None,
            *args, **kwargs
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : `TextFieldTensors`
            From a `TextField`
        label : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`

        # Returns

        An output dictionary consisting of:

            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                unnormalized log probabilities of the label.
            - `probs` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                probabilities of the label.
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits1 = self._classification_layer1(embedded_text)
        input_clf2 = torch.cat([embedded_text, logits1], dim=-1) if self._with_connected_outputs else embedded_text
        logits2 = self._classification_layer2(input_clf2)
        input_clf3 = torch.cat([embedded_text, logits2], dim=-1) if self._with_connected_outputs else embedded_text
        logits3 = self._classification_layer3(input_clf3)
        input_clf4 = torch.cat([embedded_text, logits3], dim=-1) if self._with_connected_outputs else embedded_text
        logits4 = self._classification_layer4(input_clf4)
        input_clf5 = torch.cat([embedded_text, logits4], dim=-1) if self._with_connected_outputs else embedded_text
        logits5 = self._classification_layer5(input_clf5)

        probs1 = torch.nn.functional.softmax(logits1, dim=-1)
        probs2 = torch.nn.functional.softmax(logits2, dim=-1)
        probs3 = torch.nn.functional.softmax(logits3, dim=-1)
        probs4 = torch.nn.functional.softmax(logits4, dim=-1)
        probs5 = torch.nn.functional.softmax(logits5, dim=-1)

        output_dict = {"logits1": logits1, "probs1": probs1,
                       "logits2": logits2, "probs2": probs2,
                       "logits3": logits3, "probs3": probs3,
                       "logits4": logits4, "probs4": probs4,
                       "logits5": logits5, "probs5": probs5}

        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)

        if label1 is not None or self.training:
            # For training
            output_dict["loss"] = self.loss([logits1, logits2, logits3, logits4, logits5],
                                            [label1, label2, label3, label4, label5])
        else:
            # Prediction only
            output_dict["text_repr"] = embedded_text

        return output_dict

    def _human_readable(self, predictions, lvl_label_namespace):
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(lvl_label_namespace).get(label_idx, str(label_idx))
            classes.append(label_str)

        return classes

    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """

        output_dict["label1"] = self._human_readable(output_dict["probs1"], self._label_namespace1)
        output_dict["label2"] = self._human_readable(output_dict["probs2"], self._label_namespace2)
        output_dict["label3"] = self._human_readable(output_dict["probs3"], self._label_namespace3)
        output_dict["label4"] = self._human_readable(output_dict["probs4"], self._label_namespace4)
        output_dict["label5"] = self._human_readable(output_dict["probs5"], self._label_namespace5)

        keys = [key for key in output_dict.keys() if key.startswith('logits')]
        for key in keys:
            del output_dict[key]

        keys = [key for key in output_dict.keys() if key.startswith('probs')]
        for key in keys:
            output_dict[key] = torch.round(output_dict[key].detach().cpu(), decimals=6)

        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {f'f1_{i + 1}': self.metrics[f'f1_{i + 1}'].get_metric(reset=reset)['fscore'] for i in range(5)}
        metrics['f1_avg'] = sum([metrics[f'f1_{i + 1}'] for i in range(5)]) / 5
        return metrics

    default_predictor = "five_outputs"

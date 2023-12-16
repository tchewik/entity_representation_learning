from typing import List, Dict

import numpy
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.fields import LabelField, ArrayField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.predictors.predictor import Predictor


@Predictor.register("five_outputs", exist_ok=True)
class FiveOutputsTextClassifierPredictor(Predictor):
    """
    Predictor for the model with five outputs (./five_outputs_clf.py)
    Registered as a `Predictor` with name "five_outputs".

    Allows for the encoder pooled text representation outputs.
    """

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model, and adds the `"label"` to the output.
        """
        sentence = json_dict["sentence"]
        reader_has_tokenizer = (
                getattr(self._dataset_reader, "tokenizer", None) is not None
                or getattr(self._dataset_reader, "_tokenizer", None) is not None
        )
        if not reader_has_tokenizer:
            tokenizer = SpacyTokenizer()
            sentence = tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(sentence)

    def predictions_to_labeled_instances(
            self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        label1 = numpy.argmax(outputs["probs1"])
        new_instance.add_field("label1", LabelField(int(label1), skip_indexing=True))
        label2 = numpy.argmax(outputs["probs2"])
        new_instance.add_field("label2", LabelField(int(label2), skip_indexing=True))
        label3 = numpy.argmax(outputs["probs3"])
        new_instance.add_field("label3", LabelField(int(label3), skip_indexing=True))
        label4 = numpy.argmax(outputs["probs4"])
        new_instance.add_field("label4", LabelField(int(label4), skip_indexing=True))
        label5 = numpy.argmax(outputs["probs5"])
        new_instance.add_field("label5", LabelField(int(label5), skip_indexing=True))

        new_instance.add_field("text_repr", ArrayField(outputs["text_repr"]))
        return [new_instance]

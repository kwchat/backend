import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

# define a text embedding model
X = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer("https://tfhub.dev/jeongukjae/klue_bert_cased_preprocess/1")
net = preprocessor(X)

encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/klue_bert_cased_L-12_H-768_A-12/1", trainable=True)
encoder_outputs = encoder(net)
pooled_output = encoder_outputs["pooled_output"]      # [batch_size, 768].
sequence_output = encoder_outputs["sequence_output"]  # [batch_size, seq_length, 768].

bert_pooled = tf.keras.Model(X, pooled_output)
bert_sequence = tf.keras.Model(X, sequence_output)

# define a text decoder
X = tf.keras.layers.Input(shape=(128, 768), dtype=tf.float32)
decoder = None

# define a sentence intent classifier
text_classifier = tf.keras.Sequential([
    bert_pooled,
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(1, activation='sigmoid'),
], 'intent-classifier')

# define a QA interface
class QA(tf.keras.Model):
    def __init__(self):
        super(QA, self).__init__()
        self.encoder = bert_sequence
        self.decoder = decoder

# exposing models are below
class DocRetrieveModel(tf.keras.Model):
    def __init__(self):
        super(DocRetrieveModel, self).__init__()

    def call(self, bert_code):
        pass

class DocReadModel(tf.keras.Model):
    def __init__(self):
        super(DocReadModel, self).__init__()

    def call(self, bert_code):
        pass

class DrqaModel(QA):
    def __init__(self):
        super().__init__()
        self.docRetriever = DocRetrieveModel()
        self.docReader = DocReadModel()

class DialogModel(QA):
    def __init__(self):
        super().__init__()

# final composed chatting model
class ChatModel(QA):
    def __init__(self):
        super().__init__()
        self.classifier = text_classifier
        self.chatter = DialogModel()
        self.drQA = DrqaModel()

    def call(self, text):
        intent = self.classifier(text)

        answer = tf.cond(intent < tf.constant(0.5, dtype=tf.float32),
            lambda: self.chatter(text), lambda: self.drQA(text))

        return answer

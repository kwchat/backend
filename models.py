import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

### 모든 모델이 한국어 기준으로 작성되었음 ###

# define a text embedding model
X = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer("https://tfhub.dev/jeongukjae/klue_bert_cased_preprocess/1")
preprocessor.trainable = False
net = preprocessor(X)

encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/klue_bert_cased_L-12_H-768_A-12/1", trainable=True)
encoder.trainable = True
encoder_outputs = encoder(net)
pooled_output = encoder_outputs["pooled_output"]      # [batch_size, 768].
sequence_output = encoder_outputs["sequence_output"]  # [batch_size, seq_length, 768].

embedding_size = 768
seq_length = 128

# define a text decoder
net = tf.keras.layers.Input(shape=(seq_length, embedding_size), dtype=tf.float32)
decoder = tf.keras.Sequential([], 'decoder')

# define a sentence intent classifier
text_classifier = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(embedding_size,), dtype=tf.float32),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid'),
], 'text-classifier')

# define a dialog model
dialoger = tf.keras.Sequential([], 'dialog-model')

# define a document retriever
docRetriever = tf.keras.Sequential([], 'doc-retriever')

# define a document reader
docReader = tf.keras.Sequential([], 'doc-reader')

# define a DrQA
drQA = None # to be implemented

# define a QA interface
class QA(tf.keras.Model):
    def __init__(self):
        super(QA, self).__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder

# models below are for exposing
# DrQA exam model
class DrqaModel(QA):
    def __init__(self):
        super().__init__()
        self.drQA = drQA

    def call(self, text):
        text = self.preprocessor(text)
        encoded_outputs = self.encoder(text)
        net = encoded_outputs['seqence_output']
        net = self.drQA(net)
        return self.decoder(net)

# final composed chatting model
class ChatModel(QA):
    def __init__(self):
        super().__init__()
        self.classifier = text_classifier
        self.chatter = dialoger
        self.drQA = drQA

    def call(self, text):
        text = self.preprocessor(text)
        encoded_outputs = self.encoder(text)
        pooled_output = encoded_outputs['pooled_output']
        sequence_output = encoded_outputs['sequence_output']
        intent = self.classifier(pooled_output)

        answer = tf.cond(intent < tf.constant(0.5, dtype=tf.float32),
            lambda: self.chatter(sequence_output), lambda: self.drQA(sequence_output))
        answer = self.decoder(answer)
        intent = tf.math.round(intent)

        return [answer, intent]

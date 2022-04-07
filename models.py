import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

### 모든 모델이 한국어 기준으로 작성되었음 ###

## define a text embedding model ##
preprocessor = hub.KerasLayer("https://tfhub.dev/jeongukjae/klue_bert_cased_preprocess/1")
encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/klue_bert_cased_L-12_H-768_A-12/1", trainable=True)

embedding_size = 768
max_seq_length = 128

input_shape = tf.keras.layers.Input(shape=(), dtype=tf.string)
code_shape = tf.keras.layers.Input(shape=(None, embedding_size), dtype=tf.float32)
output_shape = tf.keras.layers.Input(shape=(), dtype=tf.string)

# get encode tensors
def do_encode(text):
    text_processed = preprocessor(text)
    encoder_outputs = encoder(text_processed)
    pooled_output = encoder_outputs["pooled_output"]
    sequence_output = encoder_outputs["sequence_output"]
    reduced_sequence = tf.boolean_mask(sequence_output, text_processed['input_mask'])

    return (text_processed, encoder_outputs, pooled_output, sequence_output, reduced_sequence)

text_processed, encoder_outputs, pooled_output, sequence_output, reduced_sequence = do_encode(input_shape)

## define a text decoder ##
decoder = tf.keras.Sequential([], 'decoder')

## define a sentence intent classifier ##
text_classifier = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(embedding_size,), dtype=tf.float32),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid'),
], 'text-classifier')


## define a dialog model ##
dialoger = tf.keras.Sequential([], 'dialog-model')


## define a document retriever ##
docRetriever = tf.keras.Sequential([], 'doc-retriever')


## define a document reader ##
docReader = tf.keras.Sequential([], 'doc-reader')


## define a DrQA ##
drQA = None # to be implemented

# define a QA interface
class QA(tf.keras.Model):
    def __init__(self):
        super(QA, self).__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder


## models below are for exposing ##
# DrQA exam model
class DrqaModel(QA):
    def __init__(self):
        super().__init__()
        self.drQA = drQA

    def call(self, text):
        _, _, _, sequence_output, _ = do_encode(text)
        answer = self.drQA(sequence_output)
        answer = self.decoder(answer)
        return answer

# final composed chatting model
class ChatModel(QA):
    def __init__(self):
        super().__init__()
        self.classifier = text_classifier
        self.chatter = dialoger
        self.drQA = drQA

    def call(self, text):
        _, _, pooled_output, sequence_output, _ = do_encode(text)
        intent = self.classifier(pooled_output)

        answer = tf.cond(intent < tf.constant(0.5, dtype=tf.float32),
            lambda: self.chatter(sequence_output), lambda: self.drQA(sequence_output))
        answer = self.decoder(answer)
        intent = tf.math.round(intent)

        return [answer, intent]

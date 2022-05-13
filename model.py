import tensorflow as tf
import transformers

class BertBasedModel(tf.keras.Model):
    def __init__(self, max_length, bert_version="albert"):
        super().__init__()
        # self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        # self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.bert_version = bert_version
        self.input_ids = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="input_ids"
        )
        # Attention masks indicates to the model which tokens should be attended to.
        self.attention_masks = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="attention_masks"
        )
        # Token type ids are binary masks identifying different sequences in the model.
        if bert_version != "distillbert":
                self.token_type_ids = tf.keras.layers.Input(
                    shape=(max_length,), dtype=tf.int32, name="token_type_ids"
                )
        self.bertType_model = transformers.TFAlbertModel.from_pretrained("albert-base-v2")
        self.bertType_model.trainable = False
        
        # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))

        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling1D()

        self.dropout = tf.keras.layers.Dropout(0.3)
        self.output = tf.keras.layers.Dense(1, activation=None)


    def call(self, inputs):
        # x = self.dense1(inputs)
        # return self.dense2(x)
        bertEmbeddings = self.albert_model(
                                    self.input_ids, 
                                    attention_mask=self.attention_masks, 
                                    token_type_ids=self.token_type_ids)

        sequence_output = bertEmbeddings.last_hidden_state
        if self.bert_version != "distillbert":
            pooled_output = bertEmbeddings.pooler_output
        
        biLSTM_output = self.bi_lstm(sequence_output)
        avgPool_output, maxPool_output = self.avg_pool(biLSTM_output), self.max_pool(biLSTM_output)
        pooling_output = tf.keras.layers.concatenate( [avgPool_output, maxPool_output] )

        output = self.output(self.dropout(pooling_output))
        return output
        
        
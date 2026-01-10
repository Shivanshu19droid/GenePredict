import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from keras.saving import register_keras_serializable
from preprocessing import preprocess_input
import pickle

# ----------- Load Label Encoder ------------
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ----------- Custom Focal Loss ------------

def focal_loss(gamma=2., alpha=0.25):
    @register_keras_serializable()
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true_one_hot * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

# ----------- Custom Attention Layer ------------
@register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def call(self, inputs):
        score = tf.nn.tanh(inputs)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        return tf.reduce_sum(context_vector, axis=1)

    def get_config(self):
        return super().get_config()

# ----------- Load Trained Model ------------
model = load_model(
    "model/trained_model_2.keras",
    # custom_objects={
    #     "focal_loss": focal_loss(),
    #     "Attention": Attention
    # }
)

# ----------- Prediction Function ------------
def predict_top_3(gene_id, associated_genes, related_genes):
    processed_input = preprocess_input(gene_id, associated_genes, related_genes)

    print("Combined Input:", f"{gene_id} {associated_genes} {related_genes}")
    print("Processed Sequence:", processed_input)

    preds = model.predict(processed_input)[0]

    top_indices = preds.argsort()[-3:][::-1]
    top_probs = preds[top_indices]
    top_diseases = [label_encoder.inverse_transform([i])[0] for i in top_indices]

    return list(zip(top_diseases, top_probs))

# ----------- Test (Optional) ------------
if __name__ == "__main__":
    result = predict_top_3("51524", "TMEM138", "C3280906")
    print("\n🔍 Top 3 Predicted Diseases:")
    for disease, prob in result:
        print(f"{disease}: {prob:.2%}")

if __name__ == "__main__":
    result2 = predict_top_3("5593", "PRKG2", "C5562030")
    print("\n🔍 Top 3 Predicted Diseases:")
    for disease, prob in result2:
        print(f"{disease}: {prob:.2%}")




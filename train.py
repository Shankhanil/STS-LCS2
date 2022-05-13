import tensorflow as tf
from model import model

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)
history = model.fit(
    train_data,
    # validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    callbacks = [callback],
    workers=-1,
)
import model
import utils
import data_loader
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    data = utils.clean_data(True)
    data = utils.shuffle(data, 10000000)
    gen_train = data_loader.DataGenerator(data[:int(len(data) * 8)], moves_per_game=12)
    gen_val = data_loader.DataGenerator(data[int(len(data) * 0.8):int(len(data) * 0.9)], moves_per_game=1)
    gen_test = data_loader.DataGenerator(data[int(len(data)*0.9):], moves_per_game=1)
    xadrezia = model.Xadrezia()
    xadrezia.predict(np.random.randn(1, 8, 8, 7))
    #xadrezia.load_weights('xadrezia_weights_1.weights.h5')
    optimizer = tf.keras.optimizers.Adam(learning_rate=8e-5)
    xadrezia.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])
    xadrezia.summary()
    xadrezia.fit(gen_train, validation_data=gen_val, epochs=2)
    xadrezia.save_weights('xadrezia_weights_1.weights.h5')

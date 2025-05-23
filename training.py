from model import TransformerXadrezia
import utils
from data_loader import *

if __name__ == '__main__':
    data = utils.clean_data(True)
    data = utils.shuffle(data, 10000000)
    gen_train = DataGenerator(data[:int(len(data) * 8)], moves_per_game=16)
    gen_val = DataGenerator(data[int(len(data) * 0.8):int(len(data) * 0.9)], moves_per_game=1)
    gen_test = DataGenerator(data[int(len(data)*0.9):], moves_per_game=1)
    xadrezia = TransformerXadrezia()
    xadrezia.predict(np.random.randn(2, 8, 8, 13))
    xadrezia.summary()
    #xadrezia.load_weights('xadrezia_weights_1.weights.h5')
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    xadrezia.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])
    xadrezia.fit(gen_train, validation_data=gen_val, epochs=2)
    xadrezia.save_weights('xadrezia_weights_1.weights.h5')

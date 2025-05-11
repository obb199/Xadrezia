import model
import utils
import keras
import data_loader
import numpy as np

if __name__ == '__main__':
    data = utils.clean_data(True)
    #data = utils.shuffle(data, 10000000)
    gen_train = data_loader.DataGenerator(data[:int(len(data) * 0.8)], moves_per_game=10)
    gen_val = data_loader.DataGenerator(data[int(len(data) * 0.8):int(len(data) * 0.9)], moves_per_game=1)
    #gen_test = data_loader.DataGenerator(data[int(len(data)*0.9):], moves_per_game=2)
    try:
        xadrezia = keras.models.load_model(
            'xadrezia_model.keras',
            custom_objects={
                'Xadrezia': model.Xadrezia,
                'MultiHeadAttention': model.MultiHeadAttention,
                'Encoder': model.Encoder,
                'ResidualConvolution': model.ResidualConvolution,
                'WeightedAverage': model.WeightedAverage
            }
        )
    except ValueError:
        xadrezia = model.Xadrezia()
        xadrezia.predict(np.random.randn(1, 8, 8, 7), verbose=0)

    xadrezia.summary()
    optimizer = keras.optimizers.Adam(learning_rate=7e-5)
    xadrezia.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])
    xadrezia.fit(gen_train, epochs=5, validation_data=gen_val)
    xadrezia.save('xadrezia_model.keras')

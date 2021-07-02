import os
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf


class CaptionModel:
    WIDTH = 299
    HEIGHT = 299
    OUTPUT_DIM = 2048
    START = "startseq"
    STOP = "endseq"
    MAX_LENGTH = 34

    def __init__(self):
        # Ruta del proyecto
        root_path = os.path.dirname(os.path.abspath(__file__))
        config = tf.ConfigProto()
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession(config=config)
        self.graph = tf.get_default_graph()
        tf.keras.backend.set_session(self.sess)

        # Load the Inception model to preprocess images
        encode_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')
        self.encode_model = tf.keras.models.Model(encode_model.input, encode_model.layers[-2].output)
        self.preprocess_input = tf.keras.applications.inception_v3.preprocess_input

        # Load the pretrained model to generate the captions
        model_path = os.path.join(root_path, "data", f'caption-modelbueno.h5f5')
        weights_path = os.path.join(root_path, "data", f'caption-model.hdf5')
        self.caption_model = tf.keras.models.load_model(model_path)
        self.caption_model.load_weights(weights_path)

        # Load the vocabulary
        test_path = os.path.join(root_path, "data", f'w2id.pkl')
        with open(test_path, "rb") as path:
            self.wordtoidx = pickle.load(path)
        test_path = os.path.join(root_path, "data", f'id2w.pkl')
        with open(test_path, "rb") as path:
            self.idxtoword = pickle.load(path)

    def encode_image(self, img):
        with self.graph.as_default():
            tf.keras.backend.set_session(self.sess)
            # Resize all images to a standard size (specified bythe image encoding network)
            img = img.resize((self.WIDTH, self.HEIGHT), Image.ANTIALIAS)
            # Convert a PIL image to a numpy array
            x = tf.keras.preprocessing.image.img_to_array(img)
            # Expand to 2D array
            x = np.expand_dims(x, axis=0)
            # Perform any preprocessing needed by InceptionV3 or others
            x = self.preprocess_input(x)
            # Call InceptionV3 (or other) to extract the smaller feature set for the image.
            x = self.encode_model.predict(x)  # Get the encoding vector for the image
            # Shape to correct form to be accepted by LSTM captioning network.
            x = np.reshape(x, self.OUTPUT_DIM)
            return x

    #Generate caption
    def generate_caption(self, photo):
        with self.graph.as_default():
            tf.keras.backend.set_session(self.sess)
            in_text = self.START
            for i in range(self.MAX_LENGTH):
                #mientras no devuelva el token o la descripción sea menor de X caracteres
                sequence = [self.wordtoidx[w] for w in in_text.split() if w in self.wordtoidx]
                sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=self.MAX_LENGTH)
                y_hat = self.caption_model.predict([photo, sequence], verbose=0)  # verbose =0 dont show progress bar
                y_hat = np.argmax(y_hat)
                word = self.idxtoword[y_hat]
                in_text += ' ' + word
                if word == self.STOP:
                    break
            final = in_text.split()
            final = final[1:-1]
            final = ' '.join(final)
            return final

    def captioning(self, image_in):
        img_local = Image.open(image_in)
        img_local = self.encode_image(img_local)
        img_local = img_local.reshape((1, self.OUTPUT_DIM))
        caption = self.generate_caption(img_local)
        return caption

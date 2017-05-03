import os
from keras.utils import plot_model
from keras.models import load_model
from keras.callbacks import TensorBoard
def main():

	model_path = os.path.join('./','model.h5')
	emotion_classifier = load_model(model_path)
	emotion_classifier.summary()
	plot_model(emotion_classifier,to_file='model.png')

if __name__ == "__main__":
    main()
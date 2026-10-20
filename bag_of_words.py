import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import MeCab
import numpy as np

from massive import load

keras.utils.set_random_seed(0)
wakati = MeCab.Tagger("-Owakati")


def mecab(text):
    return wakati.parse(text).strip()


def run(name, locale, split="whitespace", preprocess=None):
    data = load(locale)
    if preprocess:
        data = {k: ([preprocess(t) for t in texts], intents) for k, (texts, intents) in data.items()}

    intents = sorted(set(data["train"][1]))
    label = {intent: i for i, intent in enumerate(intents)}

    vectorizer = keras.layers.TextVectorization(max_tokens=10_000, split=split, output_mode="multi_hot")
    vectorizer.adapt(data["train"][0])

    def xy(part):
        texts, y = data[part]
        return vectorizer(texts), np.array([label[i] for i in y])

    x_train, y_train = xy("train")
    x_dev, y_dev = xy("dev")
    x_test, y_test = xy("test")

    model = keras.models.Sequential([
        keras.layers.Input(shape=(x_train.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(len(intents), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_data=(x_dev, y_dev), epochs=10, batch_size=32, verbose=0)

    accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    print(f"{name:28s} vocabulary {len(vectorizer.get_vocabulary()):6d}   test accuracy {accuracy:.3f}")


run("French, spaces", "fr-FR")
run("Japanese, spaces", "ja-JP")
run("Japanese, characters", "ja-JP", split="character")
run("Japanese, MeCab", "ja-JP", preprocess=mecab)

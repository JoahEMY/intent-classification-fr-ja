# Intent classification: French vs Japanese

Follow-up to [tokenization-fr-ja](https://github.com/JoahEMY/tokenization-fr-ja).
That repository compared how French and Japanese text is cut into tokens. This
one checks whether the choice of tokenization matters once a model is trained
on the text.

The task is intent classification on [MASSIVE](https://github.com/alexa/massive):
short requests to a voice assistant ("wake me up at five", "what's the weather
tomorrow"), each labeled with one of 60 intents. The same sentences exist in
French and in Japanese, so both languages are compared on the same content.

## Bag of words

`bag_of_words.py` trains the same small model four times:

- French, split on spaces
- Japanese, split on spaces
- Japanese, split by character
- Japanese, split by MeCab

Each sentence becomes a multi-hot vector (`TextVectorization`,
`output_mode="multi_hot"`, 10,000 tokens max), followed by a dense layer of 64
units and a softmax over the 60 intents. The script prints the vocabulary size
and the accuracy on the test set.

## Running

```sh
pip install -r requirements.txt
python bag_of_words.py
```

The MASSIVE archive (about 40 MB) is downloaded to `data/` on the first run.

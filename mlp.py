import numpy as np
import tensorflow as tf
from tensorflow import keras

import argparse
import random
import os


# -----------------------------------------------------------------------------
# Multi Layer Perceptron
class MLP(keras.Model):

    def __init__(self, block_size, vocab_size, n_embd, n_embd2):
        """
        block_size: int, number of characters in context
        vocab_size: size of vocabulary, excluding the padding character at the start
        of the sequence
        n_embd: int,  size of the embedding
        n_embd2: int, size of the hidden layer
        """
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_embd2 = n_embd2
        # Token embedding table
        # +1 in the line below for a special <BLANK> token that gets inserted if encoding a token
        # before the beginning of the input sequence
        self.wte = keras.layers.Embedding(vocab_size + 1, n_embd)

        self.mlp = keras.models.Sequential()
        self.mlp.add(keras.layers.Dense(
                input_shape=(self.block_size * n_embd,),
                units=n_embd2,
                activation='tanh'))
        self.mlp.add(keras.layers.Dense(
            units=self.vocab_size + 1, 
            name='logits'))
        self.mlp.add(keras.layers.Softmax())

    # Annotation needed so that function is available in saved model
    @tf.function(input_signature=[])
    def get_block_size(self):
        return self.block_size

    def call(self, idx):
        # gather the word embeddings of the previous block_size words
        embs = self.wte(idx)

        # reshape the logits so that they match the input of the first dense layer
        embs = tf.reshape(embs, shape=(-1, self.block_size * self.n_embd))
        logits = self.mlp(embs)

        return logits

def get_dataset_for_mlp(idxs, block_size):
    """
    idxs: list of list of integers, each inner list represents a word
    block_size: integer

    returns: tuple of tensors (X, y). 
        X will have shape (num_examples, block_size)
        y will have shape (num_examples,)
    """
    X, y  = [], []
    for w in idxs:
        w = w + [0] # add stop token
        context = [0] * block_size
        for c in w:
            X.append(context)
            y.append(c)
            context = context[1:] + [c]
    
    return tf.constant(X), tf.constant(y)

# -----------------------------------------------------------------------------
# Sample from the model

def sample(model, num_samples, itos):
    block_size = model.get_block_size()
    if isinstance(block_size, tf.Tensor):
        block_size = block_size.numpy() # Pull out integer from tensor
    words = []
    rng = np.random.default_rng()
    for _ in range(num_samples):
        word = []
        context = [0] * block_size 
        while True:
            probs = model.call(tf.reshape(tf.constant(context), shape=(1,-1)))
            # Use numpy to sample from multinomial distribution.
            # Tensorflow expects the logits and we added a softmax layer,
            # so we have probabilities.
            ix = np.argmax(rng.multinomial(1, probs))        

            # ix = tf.random.categorical(logits, num_samples=1).numpy()[0,0]
            if ix == 0:
                break
            word.append(itos[ix])
            context = context[1:] + [ix]
            
        words.append("".join(word))
    
    return words

# -----------------------------------------------------------------------------
# Main method

if __name__ == "__main__":

    # parse command line args
    parser = argparse.ArgumentParser(description="Make More TF MLP Model")
    # system/input/output
    parser.add_argument('--input-file', '-i', 
                        type=str, default='names.txt', 
                        help="input file with things one per line")
    parser.add_argument('--work-dir', '-o', 
                        type=str, default='out', 
                        help="output working directory")

    # block_size
    parser.add_argument('--block-size',
                        type=int, default=3,
                        help="length of the context to predict next character")
    
    # model
    parser.add_argument('--n-embd', 
                        type=int, default=64, 
                        help="size of the embedding vector")
    parser.add_argument('--n-embd2', 
                        type=int, default=64, 
                        help="size of the hidden layer")

    # optimizer
    parser.add_argument('--batch-size',
                        type=int, default=64,
                        help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', 
                        type=float, default=5e-4,
                        help="learning rate")
    parser.add_argument('--patience',
                        type=int, default=5,
                        help="patience to use with early stopping callback")

    args = parser.parse_args()



    # -------------------------------------------------------------------------
    # Prepare training, validation and test data
    with open(args.input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w]
    chars = sorted(list(set(''.join(words)))) # all the possible characters

    stoi = {ch : i + 1 for i, ch in enumerate(chars)}
    itos = {i : s for s, i in stoi.items()}

    # Shuffle the list
    random.Random(42).shuffle(words)
    
    def encode(word):
        return [stoi[c] for c in word]

    # Turn words into integers
    idxs = [encode(w) for w in words]
    
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    X_train, y_train = get_dataset_for_mlp(idxs[:n1], 3)
    X_val, y_val = get_dataset_for_mlp(idxs[n1:n2], 3)
    X_test, y_test = get_dataset_for_mlp(idxs[n2:], 3)
    print(f"Training examples: {X_train.shape[0]}")
    print(f"Validation examples: {X_val.shape[0]}")
    print(f"Test examples: {X_test.shape[0]}")


    # -------------------------------------------------------------------------
    # Get and compile the model
    model = MLP(block_size=args.block_size,
                vocab_size=len(chars),
                n_embd=args.n_embd, 
                n_embd2=args.n_embd2)

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)

    # The model outputs probabilities and the labels are NOT one-hot-encoded,
    # so we use sparse categorical crossentropy as the loss function
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer)

    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=args.patience, 
        restore_best_weights=True)

    # -------------------------------------------------------------------------
    # Train, save and test the model

    model.fit(X_train, y_train, 
             epochs=10_000, # we are using the callback to stop training 
             batch_size=args.batch_size,
             validation_data=(X_val, y_val),
             callbacks=[early_stopping_cb])

    print(f"Training finished. Saving the model ....")
    print("Model summary\n")
    model.summary()
    path = os.path.join(args.work_dir, "mlp")
    model.save(path, save_format="tf")

    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, batch_size=args.batch_size)
    print("test loss:", results)


    # -------------------------------------------------------------------------
    # Sample examples from the model

    print("Sampling 20 examples from model")
    samples = sample(model, 20, itos)
    for s in samples:
        print(s)
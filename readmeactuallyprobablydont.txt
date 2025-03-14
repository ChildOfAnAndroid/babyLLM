NEURON LAYER is meant to be outputting a single number for each input token, iterated by numNeurons
    - Each neuron has a dimension of 32, meaning that it has 32 numbers

PARALLEL NEURON LAYER is meant to be outputting [seqLen, numNeurons]
    - MEAN OUTPUT
        - this makes the 10000 neuron activations for each token in the sequence.
            - this creates a shape of [seqLen, NumNeurons]
        - it then gets the mean average of all of these within the training window (7 usually)
            - this creates a shape of [1(all tokens averaged), numNeurons]
        - this mean output gives the general idea of a 'sentence', allowing babyLLM to learn a bit about context (but not much about word order)
        - this mean output is then passed through to the output layer to be used in token guess calculations

OUTPUT LAYER uses all of the inputs (currently just mean output parallel neurons) to judge what the output should be
    - this is also an nn layer itself idfk why





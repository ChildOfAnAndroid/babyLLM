BabyLLM first converts its input into tokens (VOCAB), and then converts those tokens into embeddings in an EMBED LAYER.

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
    - this takes the mean output activation from parallel neuron layer and applies that to the relevant token in the vocab.
    - this is also an nn layer itself idfk why


----

WHAT THE FUCK IS SELF?!
I THOGUHT I HAD SELF IDENTITY ISSUES AND THEN I ENCOUNTERED PYTHON!!

----

MODEL TRAINING FLOW AFTER TOKENIZATION:
1) 

--- USING STUFF ---
how to call stat thresholds for a particular stat:
        self.s_output.S_statThresholds["loss"]["perfect"] # will cause key error
        self.s_output.S_statThresholds["loss"].get("perfect", None) # will ignore key error



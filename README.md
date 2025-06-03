readmeactuallyprobablydont.txt

--- ARCHITECTURE ---

EMBED LAYER converts babyLLMs input into tokens (LIBRARIAN), and then converts those tokens into embeddings.

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

MEMORY LAYER takes the output and works it through a series of buffers/layers to figure out information from it

MEMORY LAYER 2 is a copy of the original memory layer which takes a combination of the input and the output of the first memory layer, and repasses it through its own large memory layer

LOGIT LAYER uses all of the inputs (currently from memory layer 2s output) to judge what the output should be
    - this takes the mean output activation from parallel neuron layer and applies that to the relevant token in the vocab.
    - this is also an nn layer itself idfk why

--- TRAINING ---

TUTOR

--- INFERENCE (chat) ---

INFER2

--- STATS ---

CALLIGRAPHIST

COUNSELLOR

----

what the fuck is self?! i thoguht i had self identity issues and then i encountered python!!

---

--- 
how to call stat thresholds for a particular stat:
        self.s_output.S_statBands["loss"]["perfect"] # will cause key error
        self.s_output.S_statBands["loss"].get("perfect", None) # will ignore key error



readmeactuallyprobablydont.txt

--- ARCHITECTURE ---
(adjustable) currently:
VOCAB = 4200 tokens
EMBED DIMENSION = 1024
NUM NEURONS = 10,000

EMBED LAYER converts babyLLMs input into tokens (LIBRARIAN), and then converts those tokens into embeddings.


NEURON LAYER is meant to be outputting a single number for each input token, iterated by numNeurons

    - Each neuron has a dimension of 1024, meaning that it has 1024 numbers for each tensor neuron


PARALLEL NEURON LAYER is meant to be outputting [seqLen, numNeurons]

    - WINDOWS / meaning parts of the output instead of directly using attention heads
    
        - this makes the 10000 neuron activations for each token in the sequence.
        
            - this creates a shape of [seqLen, NumNeurons]
            
        - it then gets the mean average of all tokens within the training window (256 usually)
        
            - this creates a shape of [1(all tokens averaged), numNeurons]

            - it does this 7 times, to create 7 learnable windows of different sizes
            
        - these mean outputs give the general idea of a 'sentence', allowing babyLLM to learn a bit about context, and combining multiple windows allows it to learn a tiny bit about word order.
        
        - based on learned weightings, the 7 means are then combined to create a single output to the memory layers.
        

MEMORY LAYER takes the output and works it through a series of buffers/layers to figure out information from it


MEMORY LAYER 2 is a copy of the original memory layer which takes a combination of the input and the output of the first memory layer, and repasses it through its own large memory layer


LOGIT LAYER uses all of the inputs (currently from memory layer 2s output) to judge what the output should be

    - this takes the final output activations from memory layer 2 and applies that to the relevant token in the vocab.
    
    - this is also an nn layer itself


--- TRAINING ---

TUTOR

--- INFERENCE (chat) ---

INFER2

--- STATS ---

CALLIGRAPHIST

COUNSELLOR

----

what the fuck is self?! i thought i had self identity issues and then i encountered python!!

---

how to call stat thresholds for a particular stat:
        self.s_output.S_statBands["loss"]["perfect"] # will cause key error
        self.s_output.S_statBands["loss"].get("perfect", None) # will ignore key error



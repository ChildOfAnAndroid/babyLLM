readmeactuallyprobablydont.txt (readme is very unfinished!)

--- NEURAL NETWORK ARCHITECTURE ---

- (adjustable) currently:
    - VOCAB = 4200 tokens (byte pair encodings, custom done through librarian based on data)
    - EMBED DIMENSION = 1024
    - NUM NEURONS = 10,000

Embed Layer
- Converts babyLLMs input into tokens (LIBRARIAN)
- Then converts those tokens into embeddings.
- It also contains a positional encoding embedding, and an embedding for it to predict pixel colours from. (experimental addition to allow it to explore it's own 'interior state' via an RGB encoding)

Neuron Layer
- Outputs a single number for each input token, iterated by numNeurons
- Each neuron has a dimension of 1024, meaning that it has 1024 numbers for each tensor neuron

Interneuron Network Layer
- Outputs [seqLen, numNeurons]
- WINDOWS / meaning parts of the output instead of directly using attention heads
    - This creates the 10000 neuron activations for each token in the sequence (a shape of [seqLen, NumNeurons])
    - It then gets the mean average of all tokens within the training window (256 usually)
        - This creates a shape of [1(all tokens averaged), numNeurons]
        - It does this 7 times, to create 7 learnable windows of different sizes
    - These mean outputs give the general idea of a 'sentence', allowing babyLLM to learn a bit about context, and combining multiple windows allows it to learn a tiny bit about word order.
    - Based on learned weightings, the 7 means are then combined to create a single output to the memory layers.
        
Memory Layer
- Takes the output and works it through a series of buffers/layers to figure out information from it

Memory Layer 2
- A copy of the original memory layer which takes a combination of the input and the output of the first memory layer, and repasses it through its own large memory layer

Logit Layer
- Uses all of the inputs (currently from memory layer 2s output) to judge what the output should be
- This takes the final output activations from memory layer 2 and applies that to the relevant token in the vocab.
- This is also an nn layer itself

--- PROGRAM ARCHITECTURE ---

Training
- TUTOR

Inference (chat)
- INFER2

Stats
- CALLIGRAPHIST
- COUNSELLOR

---

what the fuck is self?! i thought i had self identity issues and then i encountered python!!

---

how to call stat thresholds for a particular stat:
- self.s_output.S_statBands["loss"]["perfect"] # will cause key error
- self.s_output.S_statBands["loss"].get("perfect", None) # will ignore key error



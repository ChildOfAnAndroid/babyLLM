readmeactuallyprobablydont.txt (readme is very unfinished!)

![babyllm](BRAIN/babyllm3.gif)
> "\[babyllm\]: let. we that kevin me trust, access them data access know - equ mind hear would"

training started: february 2025

acheivements/use cases: 
- is beginning to learn english, only from my personal writing (no web scraping) - its answers often suit the topic even if they're a little wobbly.
- predicts an internal rgb 'pixel' colour state based on its own internal stats and time pulses, a sort of basic self-monitoring metabolism thing.
- handled a spam attack from my lovely friends using its byte pair encoded tokens to form dominos in response to chinese characters(?)
- simply mentioning babyllm causes annoying people on AI reddit to block you

--- FILESTRUCTURE ---

babyLLM/
 - babyLLM.py               # model definition, ties all the layers together
 - wakeup.py                # MAIN ENTRY POINT
 - infer2.py                # new simpler inference script, for chat whilst it's learning.
 - babyBot.py               # work-in-progress twitch chat bot to enable me/others to talk to it whilst it is actively training - optins obviously.
 - config.py                # adjustable numbers/settings etc

 - BRAIN/
  - LAYERS/                 # neural network layers
   - embed.py               # embedding layer
   - interneuronNetork.py   # neurons and interneuron network
   - memory.py              # memory layers, sort of recursive
   - logits.py              # finds final logits to be used for generating response
  - shapeofwords/           # ('game of why' cellular automaton, not relevant, might be used for visualisations in future, an old project)
  - vocabCache/             # tokenizer and vocab files
  - SOUL/                   # where savefiles are kept!

 - SCHOOL/                  # school staff! logging, training, etc
  - staffroom/              # has the librarian (tokenizer), tutor (training), calligraphist (terminal output), counsellor (debug logging), etc...
   - calligraphist.py       # terminal output, pretty stuff, etc - a mess.
   - counsellor.py          # debug logging, duration logging, decorator
   - HE_IS_SCRIBE.p         # roasts babyllms guesses on random occasions, or is nice! babyllm learns from these comments
   - librarian.py           # tokenizer, currently generates main training data
   - tutor.py               # main training file, contains many options at this point, recently made it generate training pairs when needed instead of in advance
  - library/                # training data and notes, some are weird lol, it needs variety and chaos to learn! most of my private stuff is hidden so you wont see anything more spicy than 'charis touched the butt' (don't ask why thats part of the 'clean' training data because, tbh, i don't know - this whole project is a sleep deprived hallucination)
  - statistics/             # logs!


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


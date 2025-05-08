from babyLLM import BABYLLM  # Adjust if needed
from SCHOOL.staffroom.counsellor import COUNSELLOR
from SCHOOL.staffroom.calligraphist import S_OUTPUT
from SCHOOL.staffroom.librarian import LIBRARIAN
from SCHOOL.staffroom.HE_IS_SCRIBE import SCRIBE
from config import *  # Should now have vocabSize = 4200

import torch
import torch.nn as nn

old_ckpt_path = "BRAIN/soul/babyLLM_2000.pth"  # your old model file
new_vocab_size = 4200

# load old checkpoint
old_state = torch.load(old_ckpt_path, map_location='cpu')

# Init new model with larger vocab
counsellor = COUNSELLOR("babyLLM", _debug = debugPrints, _durations = durationLogging)


librarian = LIBRARIAN(_counsellor = counsellor, _baseTokenizerPath = None)

calligraphist = S_OUTPUT(_counsellor = counsellor)

scribe = SCRIBE(_counsellor     = counsellor, 
                _calligraphist = calligraphist, 
                _librarian      = librarian,)  
          
model = BABYLLM(_counsellor     = counsellor,
                _calligraphist  = calligraphist, 
                _scribe         = scribe,
                _librarian      = librarian, 
                _device         = modelDevice,)

# patch embedding layer
old_embed = old_state['embed.e_weights']
new_embed = model.embed.e_weights.data
new_embed[:old_embed.size(0)] = old_embed  # copy first 2000
print(f"embedding: copied {old_embed.size(0)} to {new_embed.size(0)}")

# patch output weight
old_logits = old_state['logits.l_weights']
new_logits = model.logits.l_weights.data
new_logits[:, :old_logits.size(1)] = old_logits  # copy cols for first 2000
print(f"logits: copied {old_logits.size(1)} to {new_logits.size(1)}")

# patch output bias
old_bias = old_state['logits.l_bias']
new_bias = model.logits.l_bias.data
new_bias[:old_bias.size(0)] = old_bias
print(f"logit bias: copied {old_bias.size(0)} to {new_bias.size(0)}")

# patch logitNorm
old_norm_weight = old_state['logits.logitNorm.weight']
old_norm_bias = old_state['logits.logitNorm.bias']
new_norm_weight = model.logits.logitNorm.weight.data
new_norm_bias = model.logits.logitNorm.bias.data
new_norm_weight[:old_norm_weight.size(0)] = old_norm_weight
new_norm_bias[:old_norm_bias.size(0)] = old_norm_bias
print(f"logitNorm: copied {old_norm_weight.size(0)} to {new_norm_weight.size(0)}")

# copy everything else
for key in old_state:
    if key not in [
        'embed.e_weights',
        'logits.l_weights',
        'logits.l_bias',
        'logits.logitNorm.weight',
        'logits.logitNorm.bias'
    ]:
        model.state_dict()[key].copy_(old_state[key])

# save upgraded model
torch.save(model.state_dict(), "BRAIN/soul/babyLLM_4200.pth")
print("✅ upgraded model saved.")

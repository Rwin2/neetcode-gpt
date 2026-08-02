import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        sentences=positive+negative
        vocab=sorted({w for s in sentences for w in s.split()})
      
        word_to_id={w:i+1 for i,w in enumerate(vocab)} 
        encodings=[torch.tensor([word_to_id[w] for w in s.split()]) for s in sentences]
        return nn.utils.rnn.pad_sequence(encodings,batch_first=True)

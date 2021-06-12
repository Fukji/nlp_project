from fastai.text import *
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer


class TransformersBaseTokenizer(BaseTokenizer):
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type='bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.model_max_length
        self.model_type = model_type
        
    def __call__(self, *args, **kwargs):
        return self
    
    def tokenizer(self, t:str) -> List[str]:
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        
        tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
        return [CLS] + tokens + [SEP]


class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos=[])
        self.tokenizer = tokenizer
        
    def numericalize(self, t:Collection[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(t)
    
    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
    
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}
    
    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi: collections.defaultdict(int, {v:k for k, v in enumerate(self.itos)})


class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel, self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        logits = self.transformer(input_ids, attention_mask=attention_mask)[0]
        return logits
        

def load_model():
    path = 'app'
    learner = load_learner(path)
    return learner


def preprocess(text):
    return text

def apply(text):
    prediction = learner.predict(preprocess(text))
    result = [prediction[1].numpy().tolist()]
    result.extend(prediction[2].numpy().tolist())
    print(result)
    return result


learner = load_model()
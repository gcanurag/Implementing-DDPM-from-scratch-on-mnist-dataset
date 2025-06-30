import torch
import torch.nn as nn

class EMA():
    def __init__(self, model, decay=0.9999): # usually decay =0.9999
        self.model=model
        self.decay=decay
        self.shadow={n:p.clone().detach() for n,p in model.named_parameters() if p.requires_grad} # we take key value pairs for only trainable parameters , dont take care about frozen parameters


    def update(self):
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n]=self.shadow[n]*self.decay+p.data*(1-self.decay) # ema formula
            
    
    def apply_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n]) # It overwrites the model's actual parameter tensor p.data with the stored EMA value self.shadow[n]
        

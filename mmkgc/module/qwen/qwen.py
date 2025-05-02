from PIL import Image, ImageFile
import torch
import torch.nn as nn
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from ..BaseModule import BaseModule

class Qwen2_5_VL_4bit(BaseModule):
    """
    Wrapper around Qwen2.5-VL that:
      - Loads the backbone in 4-bit
      - Freezes all model weights
      - Exposes methods for encoding and hidden state retrieval
    """
    def __init__(
        self,
        base_ckpt="Qwen/Qwen2.5-VL-3B-Instruct",
        device="cuda"
    ):
        super().__init__()
        self.device = torch.device(device)

        # 1) Quantize to 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_ckpt,
            quantization_config=bnb_config,
            device_map="auto",
            output_hidden_states=True
        )
        # Expose the decoder (cross-modal fusion layers)
        self.decoder = self.model.model

        # 2) Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        self.hidden = self.model.config.hidden_size
        self.modality_emb = nn.Embedding(3, self.hidden)

        # 3) Processor for tokenization & vision features
        self.processor = AutoProcessor.from_pretrained(base_ckpt, use_fast=True)

    def encode(
        self,
        es: torch.Tensor,
        images: list,
        texts: list,
        max_length: int = 128
    ) -> torch.Tensor:
        """
        Encode a batch of (struct_emb, image, text) triples into a joint embedding.
        Returns: tensor of shape (B, hidden_size)
        """
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # load/normalize images
        imgs = []
        for fn in images:
            try:
                imgs.append(Image.open(fn).convert("RGB"))
            except:
                imgs.append(Image.new('RGB', (300,200), (200,200,200)))

        # tokenize + preprocess
        inputs = self.processor(
            images=imgs,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # prepend structural embedding as a soft token
        B = es.size(0)
        es = es.to(self.device)                                      # (B, D)
        input_ids = inputs.input_ids                                 # (B, S)
        attn_mask = inputs.attention_mask                            # (B, S)

        # 1) get token embeddings
        tok_emb = self.model.base_model.get_input_embeddings()(input_ids)  # (B, S, D)

        # 2) prepend es
        es_unsq = es.unsqueeze(1)                                   # (B,1,D)
        inputs_embeds = torch.cat([es_unsq, tok_emb], dim=1)        # (B, 1+S, D)

        # 3) fix attention mask
        extra = torch.ones((B,1), device=self.device, dtype=attn_mask.dtype)
        new_mask = torch.cat([extra, attn_mask], dim=1)             # (B, 1+S)

        # 4) forward pass (no grads)
        with torch.no_grad():
            out = self.model(
                inputs_embeds=inputs_embeds,
                pixel_values=inputs.pixel_values,
                attention_mask=new_mask,
                output_hidden_states=True,
                return_dict=True
            )
        # pool the first position
        return out.hidden_states[-1][:,0,:]  # (B, D)

    def get_hidden_states(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Return the last hidden states for arbitrary inputs.
        """
        inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device)
        }
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values.to(self.device)

        # no grads here either
        with torch.no_grad():
            out = self.model.base_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        return out.hidden_states[-1]  # (B, seq_len, hidden_size)
    
    def encode_pretrained(self, es, ev, et):
        """
        es, ev, et: (B, D)  — your three precomputed embeddings
        returns:   joint (B, D)
        """
        B, D = es.size()
        assert D == self.hidden

        # 1) Stack them as a length-3 “sequence”
        E = torch.stack([es, ev, et], dim=1)   # (B, 3, D)

        # 2) Add modality tags
        idx = torch.arange(3, device=E.device).unsqueeze(0)  # (1,3)
        E = E + self.modality_emb(idx)                       # (B,3,D)

        # 3) Make a full attention mask (all ones, since no padding)
        attn_mask = torch.ones(B, 3, device=E.device, dtype=torch.long)

        # 4) Run through Qwen’s encoder stack as a fusion layer
        #    The encoder expects (inputs_embeds, attention_mask)
        out = self.decoder(
            inputs_embeds=E, 
            attention_mask=attn_mask
        ).last_hidden_state  # (B, 3, D)

        # 5) Pool however you like—mean or take the first slot
        joint = out.mean(dim=1)    # (B, D)
        # or: joint = out[:,0,:]

        return joint
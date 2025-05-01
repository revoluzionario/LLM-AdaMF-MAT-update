import torch
import torch.nn as nn
from .Model import Model

from mmkgc.module.qwen.qwen import Qwen2_5_VL_4bit

class QwenAdvMixRotatE(Model):
    def __init__(
        self,
        qwen_model: Qwen2_5_VL_4bit,
        ent_tot,
        rel_tot,
        dim=100,
        margin=6.0,
        epsilon=2.0,
        img_list=None,
        text_list=None
    ):
        super(QwenAdvMixRotatE, self).__init__(ent_tot, rel_tot)
        assert img_list is not None
        assert text_list is not None
        self.img_list = img_list
        self.text_list = text_list
        self.qwen_model = qwen_model

        # margin, epsilon and ent embeddings
        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim * 2
        self.dim_r = dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )

        # structural embedding projection layer
        self.es_proj = nn.Linear(self.dim_e, self.qwen_model.hidden)

        # image embedding projection layer
        self.img_proj = nn.Linear(self.dim_e, self.qwen_model.hidden)

        # text embedding projection layer
        self.text_proj = nn.Linear(self.dim_e, self.qwen_model.hidden)   

        # joint embedding projection layer
        self.joint_proj = nn.Linear(self.qwen_model.hidden, self.dim_e)

        
    def get_joint_embeddings(self, es, images, texts):
        """
        es:        Tensor of shape (B, de) — your precomputed structural embeddings
        images:    list of PIL.Image (or pre‑tensor), length B
        texts:     list of str, length B
        returns:   joint embedding (B, de)
        """
        
        joint = self.qwen_model.encode(es, images, texts)
        return joint

    def get_joint_embeddings_fake(self, es, ev, et):
        joint = self.qwen_model.encode_pretrained(es, ev, et)
        return joint
    
    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        r = self.rel_embeddings(batch_r)

        all_ids = torch.cat([batch_h, batch_t], dim=0)
        unique_ids, inv_idx = torch.unique(all_ids, return_inverse=True)
        B = batch_h.size(0)
        head_inv = inv_idx[:B]
        tail_inv = inv_idx[B:]
        es_unique = self.es_proj(self.ent_embeddings(unique_ids))
        txt_unique = [self.text_list[i] for i in unique_ids.tolist()]
        img_unique = [self.img_list[i][0] if self.img_list[i] is not None else None for i in unique_ids.tolist()]
        joint_unique = self.get_joint_embeddings(es_unique, img_unique, txt_unique)
        head_joint = joint_unique[head_inv]
        tail_joint = joint_unique[tail_inv]
        h_joint = self.joint_proj(head_joint)
        t_joint = self.joint_proj(tail_joint)
        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        return score
    
    def _calc(self, h, t, r, mode):
        pi = self.pi_const

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1,
                               re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1,
                               re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1,
                               re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1,
                               re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(
            -1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(
            -1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

        if mode == "head_batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return score.permute(1, 0).flatten()

    def get_batch_ent_embs(self, data):
        return self.ent_embeddings(data)

    def get_fake_score(
        self,
        batch_h,
        batch_r, 
        batch_t,
        mode,
        fake_hv=None, 
        fake_tv=None,
        fake_ht=None,
        fake_tt=None
    ):
        if fake_hv is None or fake_tv is None or fake_ht is None or fake_tt is None:
            raise NotImplementedError
        
        all_ids = torch.cat([batch_h, batch_t], dim=0)
        unique_ids, inv_idx = torch.unique(all_ids, return_inverse=True)
        B = batch_h.size(0)
        head_inv = inv_idx[:B]
        tail_inv = inv_idx[B:]
        es_unique = self.es_proj(self.ent_embeddings(unique_ids))
        txt_unique = [self.text_list[i] for i in unique_ids.tolist()]
        img_unique = [self.img_list[i][0] if self.img_list[i] is not None else None for i in unique_ids.tolist()]
        joint_unique = self.get_joint_embeddings(es_unique, img_unique, txt_unique)
        head_joint = joint_unique[head_inv]
        tail_joint = joint_unique[tail_inv]
        h_joint = self.joint_proj(head_joint)
        t_joint = self.joint_proj(tail_joint)
        
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        h = self.es_proj(h)
        t = self.es_proj(t)
        fake_hv = self.img_proj(fake_hv)
        fake_tv = self.img_proj(fake_tv)
        fake_ht = self.text_proj(fake_ht)
        fake_tt = self.text_proj(fake_tt)
        # the fake joint embedding
        h_fake = self.get_joint_embeddings_fake(h, fake_hv, fake_ht)
        t_fake = self.get_joint_embeddings_fake(t, fake_tv, fake_tt)
        h_fake = self.joint_proj(h_fake)
        t_fake = self.joint_proj(t_fake)
        score_h = self.margin - self._calc(h_fake, t_joint, r, mode)
        score_t = self.margin - self._calc(h_joint, t_fake, r, mode)
        score_all = self.margin - self._calc(h_fake, t_fake, r, mode)
        return [score_h, score_t, score_all], [h_joint, t_joint, h_fake, t_fake]

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

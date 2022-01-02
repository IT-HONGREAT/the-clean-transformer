import torch.nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F

from cleanformer.tensors import subsequent_mask


class Transformer(LightningModule):
    def __init__(self, hidden_size: int, ffn_size: int,
                 vocab_size: int, max_length: int,
                 pad_token_id: int, heads: int, depth: int,
                 dropout: float, lr: float):  # noqa
        super().__init__()

        # 매우 유용한 함수!! self.hparams에 모든 하이퍼파라미터를 저장한다.
        self.save_hyperparameters()

        # 학습을 해야하는 레이어 => 임베딩 테이블, 인코더, 디코더 를 학습한다!
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.layer = torch.nn.Module()
        self.encoder = Encoder(hidden_size, heads, max_length)
        self.decoder = Decoder(hidden_size, heads, max_length)

    def forward(self, src_ids: torch.LongTensor, tgt_ids: torch.Tensor,
                src_key_padding_mask: torch.Tensor,tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
        # src_ids 는 임베딩 테이블!
        # 즉 임베딩 벡터를 불러오자
        """
        src_ids : (N,L)
        tgt_ids : (N,L)
/Users/hong-in-yeong/WEB/hiy-django/costaurant/foods/templates/account/messages/password_changed.txt        """
        # 불러오기
        src = self.token_embeddings(src_ids) # (N,L) -> (N,L,H)
        tgt = self.token_embeddings(tgt_ids) # (N,L) -> (N,L,H)

        #TODO: 나중에하기

        memory = self.encoder.forward(src)  # (N,L,H) -> (N,L,H)정보만을 더해주기 때문에 차원은 그대로 유지된다
        hidden = self.decoder.forward(tgt, memory) # (N,L,H) -> (N,L,H)

        return hidden


    # 학습을 위해서 입력(인코더와 디코더) & 레이블을 인자로 받는 함수를 정의한다
    def training_step(self, batch: tuple[torch.Tensor,torch.Tensor], **kwargs) -> dict:
        # batch 속에 들어있는 것 =>
        X, Y = batch
        # X = 입력

            # 인코더의 입력
        src_ids , src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
            # 디코더의 입력
        tgt_ids , tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]


        hidden = self.forward(src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask)
        cls = self.token_embeddings.weight # classifier (|V|, H)
        logit = torch.einsum("nlh,vh->ncl",hidden, cls) # (N,L,H)*(V,H) -> (N,L,V=클래스)
        # 하지만, 공식문서의 input 값에 대한 요구조건을 따르면 (N,V,L)이 되어야 한다.
        # L이라는 길이별로 V개의 단어중에서 어떤것이 더 확률이 높은지 알아야함!!
        # N개의 데이터에 대해서 L개의 시간대 별로 V개의 단어에 대한 확률분포를 만들기 위한 값을 구한다.
        loss = F.cross_entropy(logit,Y)
        loss = loss.sum()

        return {"loss":loss}  # 다른 값을 출력하는 것을 위해서 dict 형식으로 return 한다.

# issue_1브랜치를 새로 만들었다(learn에서 분기됨)

    def predict(self, X: torch.Tensor) -> torch.tensor:

        """
        param X (N,2,2,L)
        return label_ids (N,L)
        """
        # 인코더의 입력
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        # 디코더의 입력
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]

        for time in range(0, self.hparams['max_length']-1): # 0 -> L - 2 이렇게하는 이유?
            hidden = self.forward(src_ids,tgt_ids,
                                  src_key_padding_mask,tgt_key_padding_mask)
            cls = self.token_embeddings.weight  # (|V|,H)

            # 행렬곱
            logits = torch.einsum('nlh,vh->nlv', hidden, cls)
            # 가장 로짓값이 높은 인덱스를 바로 다음 토큰으로 지정 -> greedy decoding (매우 근소한 차이더라도 무시한다)


            # 행렬에서 가장 큰수를 찾는 함수 argmax
            ids = torch.argmax(logits, dim=2)  # (N,L,V) -> (N,L)

            # [BOS] 다음에 와야하는 단어의 id(예측된 토큰)
            next_ids = ids[:, time]  # (N,L) -> (N,)
            # 다음 시간대의 토큰을 업데이트
            tgt_ids[:, time+1] = next_ids
            # 다음 시간대의 토큰은 더이상 패딩토큰이 아니므로 마스크를 0으로 열어준다.
            tgt_key_padding_mask[:, time+1] = 0

        label_ids = tgt_ids

        return label_ids

class Encoder(torch.nn.Module):

    def __init__(self, hidden_size: int, heads: int, max_length: int):
        super().__init__()

        self.multi_head_self_attention_layer = MultiHeadAttentionLayer(hidden_size, heads, max_length, masked=False)

        #TODO - ffn

    def forward(self, x:torch.Tensor):


        contexts = self.multi_head_self_attention_layer.forward(q=x, k=x, v=x) # 단어가 쓰인 문장에서 단어가 가지는 의미를 임베딩 벡터에 인코딩해준다
        # 맥락이반영된 벡터
        return contexts


class Decoder(torch.nn.Module):

    def __init__(self, hidden_size: int, heads: int, max_length: int):
        super().__init__()
        self.masked_multi_head_self_attention_layer = MultiHeadAttentionLayer(hidden_size, heads, max_length, masked=True)
        self.multi_head_encoder_decoder_attention_layer = MultiHeadAttentionLayer(hidden_size, heads, max_length, masked=False)


    def forward(self, x:torch.Tensor, memory: torch.Tensor ):
        """

        memory: (N, L, H) - 인코더의 출력 ( 한국어 문장에 대한 기억)

        """

        contexts = self.masked_multi_head_self_attention_layer.forward(q=x, k=x, v=x)
        alignments = self.multi_head_encoder_decoder_attention_layer.forward(q=contexts , k=memory , v=memory )

        return alignments
    # TODO : ffn, residual connection

class MultiHeadAttentionLayer(torch.nn.Module):

    def __init__(self, hidden_size: int, heads: int, max_length: int, masked: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads # 머리가 몇개?    symbol: h
        self.masked = masked

        assert self.hidden_size % self.heads == 0 # 조건생성 : 나머지가 생기지 않는 조건으로 설정
        self.head_size = self.hidden_size // self.heads   # symbol:s
        self.max_length = max_length

        self.linear_q = torch.nn.Linear(hidden_size,hidden_size)
        self.linear_k = torch.nn.Linear(hidden_size,hidden_size)
        self.linear_v = torch.nn.Linear(hidden_size,hidden_size)
        self.linear_o = torch.nn.Linear(hidden_size,hidden_size)

        # 상수 텐서를 register_buffer ; 아래에 마스킹을 할때 텐서를 직접 생성하게 되면 연산이 느리고, cuda error가 발생.
        self.register_buffer("subsequent_mask", subsequent_mask(max_length))


    def forward(self, q:torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        """
        q: (N,L,H)
        k: (N,L,H)
        v: (N,L,H)
        return => (N,L,H)

        """
        N, _, _ = q.size()
        q = self.linear_q(q) # (N, L, H) * (H, H) -> (N, L, H)
        k = self.linear_k(k) # (N, L, H) * (H, H) -> (N, L, H)
        v = self.linear_v(v) # (N, L, H) * (H, H) -> (N, L, H)

        # Multihead 만들기 : concat이 아닌, reshape을 통해서 머리를 나눠주는 방식으로 만들어줘야한다.

        q = q.reshape(N,self.max_length, self.heads, self.head_size)    # (N, L, H) -> (N, L, heads, H //heads)
        k = k.reshape(N,self.max_length, self.heads, self.head_size)    # (N, L, H) -> (N, L, heads, H //heads)
        v = v.reshape(N,self.max_length, self.heads, self.head_size)    # (N, L, H) -> (N, L, heads, H //heads)



        # TODO - scaled
        # 유사도 구하기 K(키)를 transpose함. 더 좋은 방법은 einsum을 이용!
        # h 차원에 대해서 벡터의 내적이 계산되고 h의 차원은 감소한다.
        # sims = torch.einsum("nqh,nkh->nqk", q, k)  # (N,L,H) * (N,L,H) -> (N,L,L) <mutl이전>

        # (N, L, heads, head_size) * (N, L, heads, head_size) -> (N, heads, L, L)
        sims = torch.einsum("nqhs,nkhs->nhqk", q, k)


        # TODO - masking(auto-regressive)

        if self.masked:
            """
            subsequent masking 이 되지않은 부분에 -inf
            masked_fill
            """

            # ( L, L) -> (1, 1, L, L) -> (N, heads, L, L)
            mask = self.subsequent_mask.reshape(1, 1, self.max_length, self.max_length)
            mask = mask.expand(N, self.heads, -1, -1)

            sims = torch.masked_fill(sims, mask == 0, value=float("-inf"))







        attentions = torch.softmax(sims, dim=3)  # (N, q의 길이 L, k의 길이 L  :: 마지막차원을 정규화 )

        # 가중평균
        # K차원에 있는 가중치를 확률분포로 가중평균을 구해준다
        # (N, heads, L, L)
        contexts = torch.einsum("nhqk,nkhs->nqhs", attentions, v)  # (N,L,L) * (N,L,H) -> (N,L,heads, head_size)

        contexts = contexts.reshape(N, L, self.hidden_size)

        # 단순한 join이므로 변경할 필요없음
        contexts = self.linear_o(contexts)  # (N,L,H) -> (N,L,H)
        return contexts
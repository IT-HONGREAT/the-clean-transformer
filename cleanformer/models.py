import torch.nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F

class Transformer(LightningModule):
    def __init__(self, hidden_size: int, ffn_size: int,
                 vocab_size: int, max_length: int,
                 pad_token_id: int, heads: int, depth: int,
                 dropout: float, lr: float):  # noqa
        super().__init__()

        #매우 유용한 함수!! self.hparams에 모든 하이퍼파라미터를 저장한다.
        self.save_hyperparameters()

        # 학습을 해야하는 레이어 => 임베딩 테이블, 인코더, 디코더 를 학습한다!
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=hidden_size)
        self.layer = torch.nn.Module()
        self.encoder = ...
        self.decoder = ...
    def forward(self, src_ids:torch.LongTensor, tgt_ids:torch.Tensor,
                src_key_padding_mask:torch.Tensor,tgt_key_padding_mask:torch.Tensor) -> torch.Tensor:
        #src_ids 는 임베딩 테이블!
        #즉 임베딩 벡터를 불러오자
        """
        src_ids : (N,L)
        tgt_ids : (N,L)
        """
        #불러오기
        src = self.token_embeddings(src_ids) #(N,L) -> (N,L,H)
        tgt = self.token_embeddings(tgt_ids) #(N,L) -> (N,L,H)

        #TODO: 나중에하기

        memory = self.encoder.forward(src)  # (N,L,H) -> (N,L,H)정보만을 더해주기 때문에 차원은 그대로 유지된다
        hidden = self.decoder.forward(tgt, memory) # (N,L,H) -> (N,L,H)

        return hidden


    #학습을 위해서 입력(인코더와 디코더) & 레이블을 인자로 받는 함수를 정의한다
    def training_step(self, batch: tuple[torch.Tensor,torch.Tensor], **kwargs) -> dict:
        #batch 속에 들어있는 것 =>
        X, Y = batch
        # X = 입력

            #인코더의 입력
        src_ids , src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
            #디코더의 입력
        tgt_ids , tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]


        hidden = self.forward(src_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask)
        cls = self.token_embeddings.weight #classifier (|V|, H)
        logit = torch.einsum("nlh,vh->ncl",hidden, cls) #(N,L,H)*(V,H) -> (N,L,V=클래스)
        # 하지만, 공식문서의 input 값에 대한 요구조건을 따르면 (N,V,L)이 되어야 한다.
        # L이라는 길이별로 V개의 단어중에서 어떤것이 더 확률이 높은지 알아야함!!
        # N개의 데이터에 대해서 L개의 시간대 별로 V개의 단어에 대한 확률분포를 만들기 위한 값을 구한다.
        loss = F.cross_entropy(logit,Y)
        loss = loss.sum()

        return {"loss":loss}  #다른 값을 출력하는 것을 위해서 dict 형식으로 return 한다.

#issue_1브랜치를 새로 만들었다(learn에서 분기됨)

    def predict(self, X: torch.Tensor) -> torch.tensor:

        """
        param X (N,2,2,L)
        return label_ids (N,L)
        """
        # 인코더의 입력
        src_ids, src_key_padding_mask = X[:, 0, 0], X[:, 0, 1]
        # 디코더의 입력
        tgt_ids, tgt_key_padding_mask = X[:, 1, 0], X[:, 1, 1]

        for time in range(0, self.hparams['max_length']-1): #0 -> L - 2 이렇게하는 이유?
            hidden = self.forward(src_ids,tgt_ids,
                                  src_key_padding_mask,tgt_key_padding_mask)
            cls = self.token_embeddings.weight  # (|V|,H)

            #행렬곱
            logits = torch.eisum('nlh,vh->nlv', hidden, cls)
            # 가장 로짓값이 높은 인덱스를 바로 다음 토큰으로 지정 -> greedy decoding (매우 근소한 차이더라도 무시한다)


            #행렬에서 가장 큰수를 찾는 함수 argmax
            ids = torch.argmax(logits,dim=2)  # (N,L,V) -> (N,L)

            #[BOS] 다음에 와야하는 단어의 id(예측된 토큰)
            next_ids = ids[:, time]  # (N,L) -> (N,)
            # 다음 시간대의 토큰을 업데이트
            tgt_ids[:, time+1] = next_ids
            # 다음 시간대의 토큰은 더이상 패딩토큰이 아니므로 마스크를 0으로 열어준다.
            tgt_key_padding_mask[:, time+1] = 0

        label_ids = tgt_ids

        return label_ids

class Encoder(torch.nn.Module):

    pass

class Decoder(torch.nn.Module):

    pass

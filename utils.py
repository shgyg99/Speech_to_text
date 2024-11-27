import math
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

seed = 8
sample_rate = 22050
clip = 1
n_mels=128
n_fft=400

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab = torch.load("vocab.pt")

transforms = nn.Sequential(
             T.Resample(orig_freq=sample_rate, new_freq=16000,),
             T.MelSpectrogram(n_mels=n_mels, n_fft=n_fft,),
                          ).requires_grad_(False)
class TransformerModel(nn.Module):

  def __init__(self, d_model, nhead, num_encoders, num_decoders, dim_feedforward,
               dropout=0.1, activation=F.relu):
    super().__init__()

    self.d_model = d_model

    # Embedding
    self.embedding = nn.Embedding(len(vocab), embedding_dim=d_model, padding_idx=0)

    # Position Encoding
    self.pos_encoder = PositionalEncoding(d_model=d_model)

    # Transformer
    self.transformer = nn.Transformer(
        d_model=d_model, nhead=nhead,
        num_encoder_layers=num_encoders, num_decoder_layers=num_decoders,
        dim_feedforward=dim_feedforward,
        dropout=dropout, activation=activation
        )

    self.init_weights()

  def init_weights(self) -> None:
      initrange = 0.1
      self.embedding.weight.data.uniform_(-initrange, initrange)

  def forward(self, src, tgt):
    tgt = self.embedding(tgt) * math.sqrt(self.d_model)

    tgt = tgt.permute(1, 0, 2)
    tgt = self.pos_encoder(tgt)

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(tgt)).to(device)
    out = self.transformer(src, tgt, tgt_mask=tgt_mask)

    return out
class CNN2DFeatureExtractor(nn.Module):

  def __init__(self, inplanes, planes):
    super().__init__()

    self.conv1 = nn.Conv2d(1, inplanes, kernel_size=11, stride=1, padding=5, bias=False)
    self.bn1 = nn.BatchNorm2d(inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=11, stride=1, padding=5, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.conv3 = nn.Conv2d(planes, planes, kernel_size=11, stride=1, padding=5, bias=False)
    self.bn3 = nn.BatchNorm2d(planes)
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    x = self.maxpool2(x)

    return x
  

class ResNetFeatureExtractor(nn.Module):

  def __init__(self, ):
    super().__init__()

    self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    module_list = list(self.model.children())[:-5]
    self.model = nn.Sequential(*module_list)

  def forward(self, src):
    src = self.model(src)
    return src
  



class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Arguments:
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
    """
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)
  


class SpeechRecognitionModel(nn.Module):

  def __init__(self,
               d_model, nhead, num_encoders, num_decoders, dim_feedforward, dropout=0.1, activation=F.relu,
               cnn_mode='simple', inplanes=32, planes=64,
               n_mels=128, n_fft=400):
    super().__init__()

    # Transform
    self.transforms = nn.Sequential(
        T.Resample(orig_freq=sample_rate, new_freq=16000),
        T.MelSpectrogram(n_mels=n_mels, n_fft=n_fft),
        # T.FrequencyMasking()
        ).requires_grad_(False)

    # Feature embedding
    self.cnn_mode = cnn_mode
    if cnn_mode == 'simple':
      self.cnn = CNN2DFeatureExtractor(inplanes=inplanes, planes=planes)
    elif cnn_mode == 'resnet':
      self.cnn = ResNetFeatureExtractor()
    else:
      raise NotImplementedError("Please select one of the simple or resnet model")

    # Transformer
    self.transformers = TransformerModel(
        d_model=d_model, nhead=nhead,
        num_encoders=num_encoders, num_decoders= num_decoders,
        dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)

    # Classifier
    self.cls = nn.Linear(d_model, len(vocab))

    self.init_weights()

  def init_weights(self) -> None:
    initrange = 0.1
    self.cls.bias.data.zero_()
    self.cls.weight.data.uniform_(-initrange, initrange)

  def forward(self, src, tgt):
    with torch.no_grad():
      src = self.transforms(src)

    if self.cnn_mode == 'resnet':
      src = src.repeat(1, 3, 1, 1)
    src = self.cnn(src)

    batch_size, num_channels, freq_bins, seq_len = src.shape
    src = src.reshape(batch_size, -1, seq_len)
    src = src.permute(2, 0, 1)

    out = self.transformers(src, tgt)
    out = out.permute(1, 0, 2)

    out = self.cls(out)

    return out


def generate(model, vocab, audio, max_seq_len = 6000):
    with torch.inference_mode():
      feat = model.transforms(audio)
      print(feat.shape)
      feat = model.cnn(feat.unsqueeze(1))
      print(f'feat shape : {feat.shape}')
      batch_size, num_channels, freq_bins, seq_len= feat.shape
      feat = feat.reshape(batch_size, -1, seq_len).permute(2, 0, 1)
      print(feat.shape)
      enc = model.transformers.transformer.encoder(feat)

      indices = [vocab['<']]
      dec = torch.LongTensor([indices]).to(device)

      for i in range(max_seq_len):
  
          dec = model.transformers.embedding(dec) * math.sqrt(model.transformers.d_model)
          dec = model.transformers.pos_encoder(dec)
          preds = model.transformers.transformer.decoder(dec, enc)
          out = model.cls(preds)

          idx = out[-1, ...].argmax().item()
          if idx == vocab['>']:
              break

          indices.append(idx)
          dec = torch.LongTensor([indices]).T.to(device)
          res = dec.T

    return ''.join(vocab.lookup_tokens(res[0].tolist()))


def preprocess(waveform, transform=transforms):
    waveform = torch.tensor(waveform, dtype=torch.float32).squeeze()
    print(waveform.shape)
    waveform = pad_sequence([waveform], padding_value=0, batch_first=True)
    print(waveform.shape)
    return waveform

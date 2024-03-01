import torch
from src import frontends

from src.models.whisper_main import ModelDimensions, Whisper, log_mel_spectrogram,Whisper_wo_prompt
from src.models.meso_net import MesoInception4
from src.commons import WHISPER_MODEL_WEIGHTS_PATH


class WhisperMesoNet(MesoInception4):
    def __init__(self, freeze_encoder, **kwargs):
        super().__init__(**kwargs)

        self.device = kwargs['device']
        checkpoint = torch.load(WHISPER_MODEL_WEIGHTS_PATH)
        dims = ModelDimensions(**checkpoint["dims"].__dict__)
        # dims.n_audio_ctx = dims.n_audio_ctx+3
        # print(dims)
        # import pdb; pdb.set_trace()
        model = Whisper(dims)
        model = model.to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.whisper_model = model
        if freeze_encoder:
            for param in self.whisper_model.parameters():
                param.requires_grad = False

    def compute_whisper_features(self, x, prompt=None):
        specs = []
        for sample in x:
            specs.append(log_mel_spectrogram(sample))
        x = torch.stack(specs)
        # prompt = torch.transpose(prompt, 1,2)
        # x = torch.cat([prompt, x], dim=2)
        x = self.whisper_model(x,prompt)
        x = x.permute(0, 2, 1)  # (bs, frames, 3 x n_lfcc)
        x = x.unsqueeze(1)  # (bs, 1, frames, 3 x n_lfcc)
        x = x.repeat(
            (1, 1, 1, 2)
        )  # (bs, 1, frames, 3 x n_lfcc) -> (bs, 1, frames, 3000)
        x = x[:,:,:,:404]
        return x

    def forward(self, x):
        # we assume that the data is correct (i.e. 30s)
        x = self.compute_whisper_features(x)
        out = self._compute_embedding(x)
        return out


class WhisperMultiFrontMesoNet(WhisperMesoNet):
    def __init__(self, freeze_encoder, **kwargs):
        super().__init__(freeze_encoder=freeze_encoder, **kwargs)
        self.frontend = frontends.get_frontend(kwargs['frontend_algorithm'])
        print(f"Using {self.frontend} frontend!")
        n_tokens = kwargs["n_token"]
        self.soft_prompt = torch.nn.Embedding(n_tokens, 384)
        init_prompt_value = self.whisper_model.encoder.conv2.weight[:,0,0].repeat(n_tokens, 1).detach().clone()
        self.soft_prompt.weight = torch.nn.parameter.Parameter(init_prompt_value)

    def forward(self, x):
        # Frontend computation
        frontend_x = self.frontend(x)
        learned_embed = self.soft_prompt.weight.repeat(x.size(0), 1, 1)
        x = self.compute_whisper_features(x, prompt=learned_embed)
        x = torch.cat([x, frontend_x], 1)
        out = self._compute_embedding(x)
        return out


class WhisperMesoNet_wo_prompt(MesoInception4):
    def __init__(self, freeze_encoder, **kwargs):
        super().__init__(**kwargs)

        self.device = kwargs['device']
        checkpoint = torch.load(WHISPER_MODEL_WEIGHTS_PATH)
        dims = ModelDimensions(**checkpoint["dims"].__dict__)
        # dims.n_audio_ctx = dims.n_audio_ctx+3
        # print(dims)
        # import pdb; pdb.set_trace()
        model = Whisper_wo_prompt(dims)
        model = model.to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.whisper_model = model
        if freeze_encoder:
            for param in self.whisper_model.parameters():
                param.requires_grad = False

    def compute_whisper_features(self, x):
        specs = []
        for sample in x:
            specs.append(log_mel_spectrogram(sample))
        x = torch.stack(specs)
        x = self.whisper_model(x)
        x = x.permute(0, 2, 1)  # (bs, frames, 3 x n_lfcc)
        x = x.unsqueeze(1)  # (bs, 1, frames, 3 x n_lfcc)
        x = x.repeat(
            (1, 1, 1, 2)
        )  # (bs, 1, frames, 3 x n_lfcc) -> (bs, 1, frames, 3000)
        x = x[:,:,:,:404]
        return x

    def forward(self, x):
        # we assume that the data is correct (i.e. 30s)
        x = self.compute_whisper_features(x)
        out = self._compute_embedding(x)
        return out


class WhisperMultiFrontMesoNet_wo_prompt(WhisperMesoNet_wo_prompt):
    def __init__(self, freeze_encoder, **kwargs):
        super().__init__(freeze_encoder=freeze_encoder, **kwargs)
        self.frontend = frontends.get_frontend(kwargs['frontend_algorithm'])
        print(f"Using {self.frontend} frontend!")

    def forward(self, x):
        # Frontend computation
        frontend_x = self.frontend(x)
        x = self.compute_whisper_features(x)
        x = torch.cat([x, frontend_x], 1)
        out = self._compute_embedding(x)
        return out



if __name__ == "__main__":
    import numpy as np

    input_channels = 1
    device = "cpu"
    classifier = WhisperMesoNet(
        input_channels=input_channels,
        freeze_encoder=True,
        fc1_dim=1024,
        device=device,
    )

    input_channels = 2
    classifier_2 = WhisperMultiFrontMesoNet(
        input_channels=input_channels,
        freeze_encoder=True,
        fc1_dim=1024,
        device=device,
        frontend_algorithm="lfcc"
    )
    x = np.random.rand(2, 30 * 16_000).astype(np.float32)
    x = torch.from_numpy(x)

    out = classifier(x)
    print(out.shape)

    out = classifier_2(x)
    print(out.shape)
import torch
from model import AudioClassifier
from process_audio import AudioUtil

model=AudioClassifier()

model.load_state_dict(torch.load("best_model.pth",map_location= torch.device("cpu")))


labels={0: 'air_conditioner',
        1: 'car_horn',
        2: 'children_playing',
        3: 'dog_bark',
        4: 'drilling',
        5: 'engine_idling',
        6: 'gun_shot',
        7: 'jackhammer',
        8: 'siren',
        9: 'street_music',}


device="cpu"

def process_audio(audio_file,sr=44100,channel=2,duration=4000,shift_pct=0.4):
    aud = AudioUtil.open(audio_file)

    reaud = AudioUtil.resample(aud, sr)
    rechan = AudioUtil.rechannel(reaud, channel)

    dur_aud = AudioUtil.pad_trunc(rechan, duration)
    shift_aud = AudioUtil.time_shift(dur_aud, shift_pct)
    sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    
    return aug_sgram

def predict_audio(audio_file):
    spg=process_audio(audio_file)
    spg=spg.unsqueeze(0)
    spg=spg.to(device)
    
    with torch.no_grad():
        out=model(spg)
        
    result=torch.argmax(out,dim=-1)
    result=result.detach().cpu().numpy().item()
    result=labels[result]

    
    return result
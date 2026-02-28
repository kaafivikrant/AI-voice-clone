import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch 
import soundfile as sf 
from qwen_tts import Qwen3TTSModel 
import subprocess

def create_ref_audio(filename, text):
    if os.path.exists(filename):
        # Check if it's a valid wav file, if not regenerate
        try:
            sf.read(filename)
            print(f"Using existing {filename}")
            return
        except:
            print(f"Existing {filename} is invalid, regenerating...")
            
    print(f"Generating reference audio using 'say' command to {filename}...")
    try:
        # macOS say command to generate wav
        subprocess.run(["say", "-o", filename, "--data-format=LEI16@24000", text], check=True)
        print("Reference audio generated.")
    except Exception as e:
        print(f"Failed to generate audio with 'say': {e}")

# Check for device
# Force CPU due to MPS limitation with large channels
device = "cpu"
dtype = torch.float32
print(f"Using device: {device} (forced due to MPS limitation)")

# if torch.backends.mps.is_available():
#     device = "mps"
#     dtype = torch.float16
# elif torch.cuda.is_available():
#     device = "cuda:0"
#     dtype = torch.bfloat16
# else:
#     device = "cpu"
#     dtype = torch.float32

print(f"Using device: {device}")

kwargs = {}
if device == "cuda:0":
    kwargs["attn_implementation"] = "flash_attention_2"
    kwargs["dtype"] = torch.bfloat16
elif device == "mps":
    kwargs["dtype"] = torch.float16
else:
    kwargs["dtype"] = torch.float32

print("Loading model...")
try:
    model = Qwen3TTSModel.from_pretrained( 
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", 
        device_map=device, 
        **kwargs
    )
except Exception as e:
    print(f"Error loading model with kwargs {kwargs}: {e}")
    print("Retrying with default settings (auto device map)...")
    # Fallback
    model = Qwen3TTSModel.from_pretrained( 
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", 
        device_map="auto", 
    )

ref_text  = "Audio VoiceRecorder is a free app to record voice and save to MP3, WAV, OGG audio file. Record voice online from Mac OS, Linux, Android, IOS, and anywhere." 
ref_audio_path = "clone.wav"
create_ref_audio(ref_audio_path, ref_text)

print("Generating voice clone...")
try:
    wavs, sr = model.generate_voice_clone( 
        text="""
Teri aankhon mein jo sapne sajaa hai,
Unmein hi mera jahaan basa hai,
Teri hasi ki jo roshni hai,
Usmein hi mera savera chhupa hai.

Tere bina adhura sa lagta hoon,
Jaise dhadkan bina dil rehta ho,
Tu paas ho toh sab kuch hai,
Warna yeh jeena bhi kya jeena ho.

(Pre-Chorus)
Tera naam likhun main hawaon pe,
Har dua mein tujhe maangu main,
Rab se bas itni si chahat hai,
Har janam tera saath paau main.

(Chorus)
Tu meri duniya, tu mera jahaan,
Tu hi hai mera armaan…
Teri baahon mein milta sukoon,
Tu hi mera aasman…

Saath tera ho toh dar kaisa,
Har mushkil ho aasan…
Tu meri duniya, tu mera jahaan ❤️

(Verse 2)
Teri baaton mein jo mithaas hai,
Woh shayad kahin aur nahi,
Tere saath jo pal guzare,
Woh kahin bhi bekaar nahi.

Tera haath mere haathon mein,
Bas itna sa khwaab hai,
Zindagi ke har mod pe,
Tu hi mera jawab hai.

(Bridge)
Agar kabhi andhera aaye,
Main tera diya ban jaaun,
Tu muskuraaye bas itna sa,
Main duniya se lad jaaun.

(Final Chorus - Soft)
Tu meri duniya, tu mera jahaan,
Tere bina sab veeran…
Meri har khushi tujhse hai,
Tu hi meri pehchaan…

Hamesha tera, sirf tera…
Mera har ek armaan ❤️
""", 
        language="Hindi", 
        ref_audio=ref_audio_path, 
        ref_text=ref_text, 
    ) 
    
    output_file = "output_voice_clone.wav"
    sf.write(output_file, wavs[0], sr) 
    print(f"Saved to {output_file}")
except Exception as e:
    import traceback
    print(f"Error during generation: {e}")
    traceback.print_exc()

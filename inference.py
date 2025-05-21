from audiobox_aesthetics import infer as aes_infer

aes_predictor = aes_infer.initialize_predictor()

batch = [{"path": "wavs/audio.wav"}]

results = aes_predictor.forward(batch)

print(results)

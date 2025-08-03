# Exampeles


## Cuts

#### Audio
```
lhotse download libritts -p dev-clean tests/data
lhotse prepare libritts -p dev-clean tests/data/LibriTTS tests/data/manifests/libritts
lhotse cut simple --force-eager \
    -r tests/data/manifests/libritts/libritts_recordings_dev-clean.jsonl.gz \
    -s tests/data/manifests/libritts/libritts_supervisions_dev-clean.jsonl.gz \
    examples/audio_cuts.jsonl.gz
```

#### Image


#### Video


#### Text

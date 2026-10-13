# GTReview voice clips

The words GTReview speaks into narrated recordings: `lesion-1.wav` to
`lesion-500.wav` say the whole phrase "lesion <n>" in one breath, as a case of
metastases can count hundreds of lesions; the single words (`lesion`, `new`,
`now`, `zero` ... `ninety`, `hundred`) put together "new lesion 7", "now
lesion 2" and the numbers above 500.  The set is
`GTReviewLib.recordlog.VOICE_CLIPS`; `Testing/test_wavfile.py` fails on a clip
missing from this folder or not in that set.

They were synthesised with [Piper TTS](https://github.com/OHF-Voice/piper1-gpl)
and its `en_US-ljspeech-high` voice, trained on the LJ Speech dataset, which is
in the public domain.  Only Piper's output is kept here, never the program or
the model.  Every phrase was checked with a speech recogniser (faster-whisper,
`small.en`) and synthesised again until its number was heard right.  They are
8-bit G.711 mu-law at 11025 Hz, about 10 MB in all.

`Packaging/make_voice_clips.py` makes them again.  To use a human voice
instead, record mono WAVs (16-bit PCM or 8-bit mu-law, any sample rate) under
the same names; a phrase clip that is missing is spoken word by word, so the
single words alone are enough to start with.

Only the `.wav` files are packaged; this note stays in the repository.

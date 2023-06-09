# DCASE-TASK1

Code for running DCASE TASK1 experiments (acoustic scene classification).

How to use the code:

1. Run `melspects.py` to extract features. Script accepts the location of the extracted DCASE data (`<data>`) and a path to store the exracted features in a folder (`<features>`)
2. Run `training.py` to train a model. Script accepts `<data>` as its `--data-root` parameter, `<features>/features.csv` as its `--features` parameter and a `--results-root` to store results in

Other parameters can be adapted (`--learning-rate`, `--batch-size` and so on).

If you want to include new models, adapt the `--approach` parameter to support them.

## DCASE-TASK1 2022

Data and features for the 2022 version have already been extracted in:

1. `<data>`: `/data/eihw-gpu5/trianand/DCASE/d22-t1/TAU-urban-acoustic-scenes-2022-mobile-development`
2. `<features>`: `/data/eihw-gpu5/trianand/DCASE/d22-t1/torchlibrosa-melspects/features.csv`

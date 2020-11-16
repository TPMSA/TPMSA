Notes:
1. Change the DATA_PATH in constants.py to the path of MOSI or MOSEI.
2. Change the VISUAL_DIM in constants.py acording to the dataset applied.
3. Set the CKPT in constants.py to the path of the ckeckpoint derived from fine-tuning BERT.
4. Set the SAVE_PATH in constants.py to any path you perfer to save the ckeckpoint and results.
5. The single Text Encoder version should be fine-tuned at first, then run the completed TPMSA model.

Quick run:
python run.py

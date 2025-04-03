# Jakob facial verification using opencv library

Learnt training machine learning models using a library, handling large quantities of files, Tweaking model parameters to get better results (.detectMultiScale), how video is captured and displayed in code, threshold tweaking. 
Could experiment a lot more with this to try to minimize EER. 

Started on further experimentation, now with multimodal verification with the adding of iris detection and verification. Not successful so far. Need a more reliable way to detect iris than cv2.Houghcircles. It picks up my nostrils and stuff. The idea is to get a lot of processed images of the irises from the raw data, and use that to train a model to recognize and give a confidence score on whether thats my iris. Further, i would calculate a final confidence score with a weighted average of the two seperate scores. AKA decision-level fusion. The hope is that his makes the whole system (jakob_verification.py) more reliable (lower the FAR and FRR). 

Thinking more in the future, after these goals are achieved, i want to do some experiments and calculate the FAR, FRR, and EER and optimize the thresholds and weights.

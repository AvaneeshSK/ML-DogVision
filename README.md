# ML-DogVision

kaggle : https://www.kaggle.com/competitions/dog-breed-identification

trained on 10,222 different images of dog

Model used : 

    ANN - 2L 
    CNN - 6L
    EfficientNet_v2_b3 (Stock)
    EfficientNet_v2_b3 (Fine tuning - top 10 layers unfreezed)
    

preprocessing : custom image preprocessing, onehotencoded labels (datasets not created)

Evaluations Done : accuracy_score, confusion_matrix

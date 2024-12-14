# Deeplearning-Project

Sentiment Analysis of movie reviews

This project aims to explore different models in the task of sentiment analysis using the dataset SST-5 (Stanford Sentiment Treebank -5) 
We use a total of 6 models as follows
- Convolutional Neural Network (CNN)
- Long-short-term-memory
- BERT
- deBERTa
- Albert
- Xlnet

To use these models for inference, you can run the following script:

## 1.Environment Setup
Clone the repository 
```
git clone <hhttps://github.com/Nhawtanhy11/Deeplearning-Project.git>
cd <Deeplearning-Project>
```

## 2.Model Checkpoints

The model checkpoints  are stored in the following google drive file which contains 6 files for 6 models
You can download the model checkpoint file from [this link](https://drive.google.com/drive/folders/1bStB5XpMiF0uwCB2WM7IkjHLEFZVGm74?usp=drive_link).
After downloading, place the file in the `./checkpoints/` directory, or adjust the path in the `--checkpoint` argument in `infer.py` to the correct path.

You can run the following command to test on the image at your working directory
```
python3 infer.py --image_path path_to_image/image.jpeg --checkpoint checkpoints/model.pth
```

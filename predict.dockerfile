FROM trainer:latest

ENTRYPOINT ["python", "-u", "src/models/predict_model.py", "evaluate", "/models/train_checkpoint.pth"]
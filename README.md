# LSS-TensorRT
모델 변환 과정: Train(pt) --> export_model_onnx(onnx) --> trtexec을 통해 trt 변환

## 모델 학습
```
python main.py trainval 
```

## 학습된 모델 onnx 변환
```
python main.py export_model_onnx mini runs/model{X}.pt # 원하는 모델 pt 파일 변환
```
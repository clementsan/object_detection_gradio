from transformers import pipeline

detector = pipeline(model="facebook/detr-resnet-50", revision="no_timm")
result = detector("http://images.cocodataset.org/val2017/000000039769.jpg")
print(result)
# x, y  are expressed relative to the top left hand corner.
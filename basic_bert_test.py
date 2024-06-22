from transformers import pipeline

model = pipeline('text-classification',model='./training_basic-bert/checkpoint-669')
print(model('非洲很难，手机充电都是问题，白天和一堆人挤着用水，床可能都没有，睡地铺'))
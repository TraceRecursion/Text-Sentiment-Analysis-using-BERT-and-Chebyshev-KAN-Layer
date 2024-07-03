from transformers import pipeline

model = pipeline('text-classification',model='./training_basic-bert3/checkpoint-6690')
print(model('把海弄干的鱼在海干前上了陆地，从一片黑暗森林奔向另一片黑暗森林。'))
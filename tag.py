from transformers import pipeline
model = pipeline('text-classification',model='./training_dir/checkpoint-669')
print(model('百分之70概率是中性的'))
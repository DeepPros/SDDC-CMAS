from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import InputExample

model = SentenceTransformer('DMetaSoul/sbert-chinese-general-v2')
train_examples = []

train_path = ""  # the path to generated, paraphrased or translated train text
output_path = ""  # the path to output fine-tuned model

# with open("gen_train.csv", "r",encoding='utf-16') as f:
with open(train_path, "r") as f:
    for line in f:
        num = int(line.strip()[0])
        sen = [line.strip()[2:].replace("\"", "")]
        train_examples.append(InputExample(texts=sen, label=num))

train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=12)
train_loss = losses.BatchAllTripletLoss(model=model)

# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=70, warmup_steps=100, output_path=output_path)

# Then use the output model to run Doc-SCAN.

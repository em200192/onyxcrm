# finetune_model.py

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from training_data import get_training_examples # Import from our new file

model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_name)

new_model_path = "./my-finetuned-erp-model"

triplets = get_training_examples()
train_examples = []
for anchor, positive, negative in triplets:
    train_examples.append(InputExample(texts=[anchor, positive, negative]))

print(f"Created {len(train_examples)} training examples.")

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.MultipleNegativesRankingLoss(model=model)

num_epochs = 4 # You can adjust this, 2-4 is usually good
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10% of training steps


from safetensors import SafetensorError
print("Starting the fine-tuning process...")
try:
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              output_path=new_model_path,
              show_progress_bar=True)
except SafetensorError:
    print("\nSafetensors failed to save. This is a known Windows issue.")
    print("The model was trained successfully. Saving manually with a compatible format...")
    # The model is already trained in memory, so we just save it manually
    # using a more compatible method that avoids the Windows error.
    model.save(new_model_path, safe_serialization=False)

print(f"\nFine-tuning complete. Your new, smarter model is saved to: {new_model_path}")
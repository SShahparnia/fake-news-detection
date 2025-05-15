import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("fake_news_dataset.csv", names=["title", "text", "date", "source", "author", "category", "label"], header=0)

counts = df["label"].value_counts()

plt.figure(figsize=(8, 6))
counts.plot(kind="bar", color=["blue", "orange"])
plt.title("Distribution of Fake and Real News")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("label_distribution.png")
plt.close()
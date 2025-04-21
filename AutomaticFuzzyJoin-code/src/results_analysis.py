import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

results_path = os.path.join(os.path.dirname(__file__), "results.csv")


# Load the CSV
df = pd.read_csv(results_path)

# Optional: set Seaborn theme
sns.set(style="whitegrid")

# Summary statistics
print(df.describe())

# 1. Precision, Recall, F1 bar plots
metrics = ["precision", "recall", "f1"]
for metric in metrics:
    plt.figure(figsize=(12, 6))
    sorted_df = df.sort_values(by=metric, ascending=False)
    sns.barplot(x="name", y=metric, data=sorted_df)
    plt.xticks(rotation=90)
    plt.title(f"{metric.capitalize()} by Entity Type")
    plt.tight_layout()
    plt.savefig(f"{metric}_barplot.png")
    plt.show()

# 2. Precision vs Recall scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="precision", y="recall", hue="f1", palette="viridis", s=100)
plt.title("Precision vs Recall")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_vs_recall.png")
plt.show()

# 3. Top 5 F1
top_f1 = df.sort_values(by="f1", ascending=False).head(5)
plt.figure(figsize=(8, 5))
sns.barplot(x="name", y="f1", data=top_f1)
plt.title("Top 5 F1 Scores")
plt.tight_layout()
plt.savefig("top5_f1.png")
plt.show()

# 4. Bottom 5 F1
bottom_f1 = df.sort_values(by="f1", ascending=True).head(5)
plt.figure(figsize=(8, 5))
sns.barplot(x="name", y="f1", data=bottom_f1)
plt.title("Bottom 5 F1 Scores")
plt.tight_layout()
plt.savefig("bottom5_f1.png")
plt.show()



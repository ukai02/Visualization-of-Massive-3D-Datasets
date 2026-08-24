import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_optimized_plot():
    csv_file = "timing_data_optimized.csv"
    output_filename = "GroupXY_Plot.png"

    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}. Please make sure it is in the same folder.")
        return

    df = pd.read_csv(csv_file)

    if 'Size' in df.columns:
        df.rename(columns={'Size': 'GridSize'}, inplace=True)

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    sns.boxplot(
        x="Processes", 
        y="Time", 
        hue="GridSize", 
        data=df, 
        palette="Set2", 
        linewidth=1.5
    )

    plt.title("Execution Time vs Process Count (Optimized)", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Processes (P)", fontsize=12)
    plt.ylabel("Execution Time (Seconds)", fontsize=12)
    plt.legend(title="Grid Size (nx=ny=nz)", loc="center right")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close() 
    print(f"Successfully generated: {output_filename}")

if __name__ == "__main__":
    create_optimized_plot()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Data
decoder_layers = [4, 3, 2, 1]
cider_scores = [0.7980, 0.4461, 0.4377, 0.4344]
rouge_l_scores = [0.3478, 0.3001, 0.28944, 0.28764]
rouge_1_scores = [0.3740, 0.3120, 0.3110, 0.3091]
colors = ['purple', 'red', 'green', 'blue']
pruning_percentages = ['346.5 MB', '256.4 MB', '240.0 MB', '218.2 MB']
marker_sizes = [20, 15, 10, 5]  # Larger marker for higher pruning percentage

# Create the plot
plt.figure(figsize=(12, 8))

# Plot ROUGE-1 scores
for i in range(len(decoder_layers)):
    plt.plot(decoder_layers[i], rouge_1_scores[i], marker='^', color=colors[i], markersize=marker_sizes[i])
    plt.text(decoder_layers[i], rouge_1_scores[i] - 0.02, pruning_percentages[i], fontsize=10, color=colors[i], ha='center')
plt.plot(decoder_layers, rouge_1_scores, linestyle='dotted', color='orange', label='ROUGE-1 Score')

# Plot ROUGE-L scores
for i in range(len(decoder_layers)):
    plt.plot(decoder_layers[i], rouge_l_scores[i], marker='o', color=colors[i], markersize=marker_sizes[i])
    plt.text(decoder_layers[i], rouge_l_scores[i] - 0.02, pruning_percentages[i], fontsize=10, color=colors[i], ha='center')
plt.plot(decoder_layers, rouge_l_scores, linestyle='dotted', color='purple', label='ROUGE-L Score')

# Plot CIDEr scores
for i in range(len(decoder_layers)):
    plt.plot(decoder_layers[i], cider_scores[i], marker='s', color=colors[i], markersize=marker_sizes[i])
    plt.text(decoder_layers[i], cider_scores[i] - 0.02, pruning_percentages[i], fontsize=10, color=colors[i], ha='center')
plt.plot(decoder_layers, cider_scores, linestyle='dotted', color='cyan', label='CIDEr Score')

# Add title and labels
plt.title('Scores vs. Number of Decoder Layers')
plt.xlabel('Number of Decoder Layers')
plt.ylabel('Score')

# Add table to indicate color coding
red_patch = mpatches.Patch(color='red', label='3 Decoder Layers')
green_patch = mpatches.Patch(color='green', label='2 Decoder Layers')
blue_patch = mpatches.Patch(color='blue', label='1 Decoder Layer')
plt.legend(handles=[red_patch, green_patch, blue_patch, 
                    mpatches.Patch(color='orange', label='ROUGE-1 Score'),
                    mpatches.Patch(color='purple', label='ROUGE-L Score'),
                    mpatches.Patch(color='cyan', label='CIDEr Score')],
           loc='upper right')

# Show the plot
plt.grid(True)
plt.gca().invert_xaxis()  # Invert the x-axis to show the layers in decreasing order
plt.tight_layout()
plt.show()

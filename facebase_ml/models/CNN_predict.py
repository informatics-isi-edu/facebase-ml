import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc
from pathlib import Path
import tensorflow as tf
import csv

def predict_with_metrics(model_path, test_dataset, output_dir):
    model = tf.keras.models.load_model(model_path)
    filenames = []
    y_true = []
    y_pred = []
    scores = []
    for i, (image_batch, label_batch) in enumerate(test_dataset):
        predictions = model.predict(image_batch).flatten()

        scores.extend(predictions)
        predicted_classes = (predictions > 0.5).astype(int)
        y_pred.extend(predicted_classes)
        y_true.extend(label_batch)

        filenames.extend([f"Image_{i*len(image_batch)+j+1}" for j in range(len(image_batch))])

    # Calculate F1 Score, ROC Curve and AUC
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score: {f1}")

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc}")

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'roc_curve.png')
    plt.close()

    with open(output_dir / "predictions_results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'True Label', 'Prediction', 'Probability Score'])

        for i in range(len(filenames)):
            writer.writerow([filenames[i], y_true[i], y_pred[i], scores[i]])

    print("Predictions and metrics saved.")
    return str(output_dir / "predictions_results.csv")

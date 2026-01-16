
```markdown
# Results

## Performance Metrics
Models were evaluated using accuracy, precision, recall, and F1-score.

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

Confusion Matrix
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()

The confusion matrix image is saved in assets/confusion_matrix.png.

Interpretation

The results indicate that machine learning models can reliably detect malicious IoT traffic in smart home environments.


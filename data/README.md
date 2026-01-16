---

# STEP 3: data/README.md (DATASET DOCUMENTATION)

📍 Location: data/README.md  
📍 Purpose: Shows dataset understanding and ethical use

```markdown
## SHIDD Dataset

The SHIDD dataset was created by Jacob (2023) and contains real IoT malware and benign network traffic captures.

### Download
(https://www.kaggle.com/datasets/bobaaayoung/dataset-invade)

### Usage in This Project
- Network flow CSV files were used
- Labels were converted to binary classes:
  - 0 = Benign
  - 1 = Malicious

### Important Note
The raw dataset is not included in this repository due to:
- Large file size
- Dataset licensing terms

Users must download and place the processed CSV file in this folder before running the code.
```

### Usage in Notebook
The following code cell is used to load the dataset:

```python
import pandas as pd
data = pd.read_csv("data/iot23_flows.csv")

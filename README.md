# Machine-Learning-for-Binary-Classification-of-Diabetes

Using machine learning to predict diabetes from health indicators (glucose, BMI, age, pregnancies, etc.).  
Compares **Decision Tree**, **Random Forest**, **Bagging Classifier**, and **Extra Trees**.

## Dataset

PIMA Indian Diabetes Dataset (binary outcome: diabetic / non‑diabetic).

## Visualizations

### Histograms
![Histogram of the Features](histogram.png)
The histograms show the distribution of each health indicator. Notice that **Glucose**, **Blood Pressure**, **Skin Thickness**, and **Insulin** have values of zero, which are physiologically impossible – these indicate missing data that will be imputed. **BMI** follows a near‑normal distribution, while **Age** and **Pregnancies** are right‑skewed (most patients are younger with few pregnancies). **Insulin** shows a large spike at zero, confirming widespread missing entries.

### Pair Plot (Glucose vs Insulin)
![Pairwise Scatter Plots](pairwise.png)
A positive trend is visible: higher glucose levels tend to accompany higher insulin levels. Two clusters appear (red and gray), possibly representing diabetic vs non‑diabetic groups. The spread indicates that other factors also influence insulin levels.

### Box Plot (outliers)
![Box plot](boxplot_all.png)
Features like **Insulin** and **Blood Pressure** have many outliers (points beyond 1.5×IQR). These extreme values can skew model training, so we will apply Robust Scaling and IQR‑based outlier treatment.

### Correlation Heatmap
![Heatmap](heatmap_corr.png)
**Glucose** (0.29), **Age** (0.23), and **BMI** (0.20) have the highest positive correlation with diabetes outcome, making them strong predictors. **Blood Pressure** (0.07) and **Skin Thickness** (0.08) show very weak correlation. The heatmap also reveals that **Glucose** and **Insulin** are moderately correlated (0.66), but not enough to cause multicollinearity problems.

## Preprocessing

- Median imputation for missing values  
  ![Missing before](missing_before.png) → ![Missing after](missing_after.png)
  Many features (Glucose, Blood Pressure, Skin Thickness, Insulin) had zero values that were treated as missing. After applying `SimpleImputer` with median strategy, all missing values are filled, preserving the overall distribution.

- Robust scaling (median + IQR)  
  ![Robust scaling](robust_scaling.png)
  RobustScaler uses median and IQR, making it insensitive to outliers. After scaling, all features have a similar range (centered around zero), which helps gradient‑based models converge faster.
  
- IQR outlier removal  
  ![IQR method](iqr_method.png)
  Any data point below Q1 – 1.5×IQR or above Q3 + 1.5×IQR is considered an outlier. We removed these extreme values to prevent them from dominating the model.
  
- RandomOverSampler + 80/20 train‑test split  
  ![Data split](data_split.jpg)
  The data is split into training (80%) and testing (20%) sets. The test set remains unseen during training to give an honest evaluation of generalization.

## Models & Confusion Matrices

| Model | Confusion Matrix |
|-------|------------------|
| Random Forest | ![RF CM](cm_rf.png) |
| Extra Trees | ![ET CM](cm_et.png) |
| Bagging Classifier | ![Bagging CM](cm_bagging.png) |
| Decision Tree | ![DT CM](cm_dt.png) |

## Performance Metrics

| Model            |  Accuracy  |  Precision  |  Recall  |  F1‑Score  |  ROC‑AUC |
|------------------|------------|-------------|----------|------------|----------|
| Extra Trees      |  0.9425    |   0.9620    |  0.9157  |  0.9383    | 0.9781   |
| Random Forest    |  0.9310    |   0.9080    |  0.9518  |  0.9294    | 0.9741   |
| Bagging          |  0.9080    |   0.9136    |  0.8916  |  0.9024    | 0.9584   |
| Decision Tree    |  0.8218    |   0.9755    |  0.8434  |  0.8187    | 0.8849   |

![Comparison chart](perf_comparison.png)
**Extra Trees** achieves the highest accuracy (94.25%) and F1‑score (0.938). **Random Forest** has the best recall (0.952), meaning it catches the most actual diabetic patients, though at the cost of slightly lower precision. **Decision Tree** lags behind all ensemble methods.

### ROC Curves
![ROC curves](roc_curves.png)
The **Extra Trees** curve is closest to the top‑left corner, with an AUC of 0.9781, indicating excellent separability between classes. Random Forest (0.9741) and Bagging (0.9584) also perform very well, while Decision Tree (0.8849) is clearly inferior. The high AUC values confirm that all ensemble models are highly capable of distinguishing diabetic from non‑diabetic patients.

## Conclusion

**Extra Trees** performs best overall. Random Forest is also strong. Decision Tree weakest. Future improvements: hyperparameter tuning, testing on larger real‑world data.

## References

- scikit‑learn documentation (Random Forest, Extra Trees, Bagging)
- Evidently AI – Precision/Recall guide
- Bhandari, P. – Interquartile Range (Scribbr)

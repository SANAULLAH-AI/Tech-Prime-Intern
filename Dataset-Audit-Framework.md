# 🚀 **COMPLETE DATASET AUDITING & CORRECTION SYSTEM**
## *Enterprise-Grade Data Quality Framework for AI/ML Projects*

---

## 📋 **TABLE OF CONTENTS**

1. [**Universal Truths & Principles**](#1-universal-truths--principles)
2. [**Dataset Type Classification**](#2-dataset-type-classification)
3. [**Audit Checklist by Dataset Type**](#3-audit-checklist-by-dataset-type)
4. [**Diagnostic Signals & Their Meanings**](#4-diagnostic-signals--their-meanings)
5. [**Risk Identification Matrix**](#5-risk-identification-matrix)
6. [**Correction Strategies & Code**](#6-correction-strategies--code)
7. [**Model Selection Guide**](#7-model-selection-guide)
8. [**Validation Framework**](#8-validation-framework)
9. [**Complete Python Implementation**](#9-complete-python-implementation)
10. [**Quick Reference Cheat Sheet**](#10-quick-reference-cheat-sheet)

---

## 1. **UNIVERSAL TRUTHS & PRINCIPLES**

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    THE GOLDEN RULES OF DATA SCIENCE                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  1. GARBAGE IN → GARBAGE OUT (GIGO)                                        ║
║  2. DATA MATTERS MORE THAN ALGORITHMS                                      ║
║  3. CORRELATION ≠ CAUSATION                                                ║
║  4. ALL MODELS ARE WRONG, SOME ARE USEFUL                                  ║
║  5. NO FREE LUNCH (No single best algorithm)                               ║
║  6. OVERFITTING IS THE ENEMY                                               ║
║  7. MORE DATA > MORE COMPLEXITY                                            ║
║  8. BIAS CANNOT BE ELIMINATED, ONLY MANAGED                                ║
║  9. INTERPRETABILITY MATTERS                                               ║
║ 10. EVALUATION DETERMINES SUCCESS                                          ║
║ 11. UNCERTAINTY IS INEVITABLE                                              ║
║ 12. DOMAIN KNOWLEDGE IS A SUPERPOWER                                       ║
║ 13. AI LEARNS PATTERNS, NOT TRUTH                                          ║
║ 14. LLMs PREDICT TOKENS, NOT MEANING                                       ║
║ 15. HUMAN FEEDBACK REMAINS ESSENTIAL                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. **DATASET TYPE CLASSIFICATION**

### **A. By Data Structure**

| **Type** | **Description** | **Examples** | **Storage Format** |
|----------|-----------------|--------------|-------------------|
| **Structured** | Organized in rows/columns | SQL databases, Excel | .csv, .xlsx, .parquet |
| **Unstructured** | No predefined format | Videos, PDFs, Audio | .mp4, .pdf, .mp3 |
| **Semi-Structured** | Has tags/markers | JSON, XML, HTML | .json, .xml, .html |

### **B. By Statistical Data Type**

| **Type** | **Description** | **Examples** | **ML Use Case** |
|----------|-----------------|--------------|-----------------|
| **Numerical - Continuous** | Any value in range | Weight, Height, Temp | Regression |
| **Numerical - Discrete** | Countable whole numbers | No. of cars, Children | Count Prediction |
| **Categorical - Nominal** | No natural order | Eye color, Country | Classification |
| **Categorical - Ordinal** | Has meaningful order | Education Level, Ratings | Ordinal Regression |
| **Binary** | Only two outcomes | Yes/No, True/False | Binary Classification |

### **C. By Media Format**

| **Type** | **Description** | **Model Architecture** | **Key Libraries** |
|----------|-----------------|----------------------|-------------------|
| **Text/Corpus** | Written language | BERT, GPT, LSTM | HuggingFace, NLTK, SpaCy |
| **Image** | Visual data | CNN, ViT, ResNet | PyTorch, TensorFlow, OpenCV |
| **Audio** | Sound recordings | Wav2Vec, Whisper | Librosa, TorchAudio |
| **Video** | Moving media | I3D, TimeSformer | OpenCV, PyTorchVideo |
| **Multimodal** | Multiple media types | CLIP, LLaVA | Transformers, CLIP |

### **D. By Time Dimension**

| **Type** | **Description** | **Example** | **Splitting Strategy** |
|----------|-----------------|-------------|----------------------|
| **Time-Series** | Sequential over time | Stock prices, ECG | Time-based split |
| **Cross-Sectional** | Single point in time | Census 2026 | Random split |
| **Longitudinal/Panel** | Same subjects over time | Medical trials | LeaveOneGroupOut |
| **Streaming/Real-Time** | Continuous live data | GPS, IoT feeds | Online learning |

### **E. By File Format**

| **Format** | **Structure** | **Best Used For** | **Data Size** |
|------------|---------------|-------------------|---------------|
| **CSV/TSV** | Tabular | Small to medium datasets | < 10GB |
| **Parquet** | Columnar storage | Large tabular datasets | 10GB - 1TB |
| **HDF5** | Hierarchical | Scientific data, huge arrays | Any size |
| **JSON/XML** | Nested keys/values | Web APIs, configuration | < 100MB |

---

## 3. **AUDIT CHECKLIST BY DATASET TYPE**

### **🔍 TEXT DATASETS (NLP, LLMs, Resume Parsing)**

```python
class TextDatasetAudit:
    """Complete audit checklist for text datasets"""
    
    def __init__(self, dataset_path):
        self.path = dataset_path
        
    def run_full_audit(self):
        """Run all text dataset checks"""
        
        # 1. FORMAT CORRUPTION TEST
        print("🔍 Testing Text Extraction Rate...")
        extraction_rate = self.check_extraction_rate()
        # ✅ PASS: >95% clean extraction
        # ❌ FAIL: <95% extraction (Scanned PDFs, corrupt files)
        
        # 2. STRUCTURAL DENSITY INDEX
        print("🔍 Measuring Structural Density...")
        length_stats = self.analyze_sequence_lengths()
        # Signals: avg_length, max_length, min_length, std_dev
        
        # 3. ENTITY SPARSITY MATRIX
        print("🔍 Checking Entity Frequency...")
        entity_counts = self.count_entities()
        # Signals: EMAIL: 90%, NAME: 95%, SKILLS: 45%, PHONE: 40%
        # ⚠️ WARNING: If any core entity < 50%, issue!
        
        # 4. OUT-OF-VOCABULARY (OOV) RATE
        print("🔍 Detecting OOV Tokens...")
        oov_ratio = self.calculate_oov_rate()
        # ✅ PASS: <10% OOV
        # ❌ FAIL: >15% OOV (Domain mismatch)
        
        # 5. DUPLICATE CONTENT RATIO
        print("🔍 Scanning for Duplicates...")
        duplicate_ratio = self.check_duplicates()
        # ✅ PASS: <5% duplicates
        # ❌ FAIL: >15% duplicates
        
        # 6. TOXICITY & PII DENSITY
        print("🔍 Scanning for PII/Toxicity...")
        pii_count = self.scan_pii()
        # ⚠️ WARNING: Any PII detected → REDACT!
        
    def check_extraction_rate(self):
        """Check PDF/image text extraction quality"""
        # Use PyMuPDF for PDFs
        # Use Tesseract OCR for scanned images
        pass
    
    def analyze_sequence_lengths(self):
        """Analyze text length distribution"""
        pass
    
    def count_entities(self):
        """Count entity occurrences using NER"""
        # Use SpaCy or HuggingFace NER
        pass
    
    def calculate_oov_rate(self):
        """Calculate Out-of-Vocabulary rate"""
        pass
    
    def check_duplicates(self):
        """Detect duplicate documents using MinHash LSH"""
        from datasketch import MinHash, MinHashLSH
        pass
    
    def scan_pii(self):
        """Scan for Personal Identifiable Information"""
        # Use Presidio Analyzer
        pass
```

---

### **🖼️ IMAGE DATASETS (CV, CNNs, Vision)**

```python
class ImageDatasetAudit:
    """Complete audit checklist for image datasets"""
    
    def __init__(self, dataset_path):
        self.path = dataset_path
        
    def run_full_audit(self):
        """Run all image dataset checks"""
        
        # 1. RESOLUTION DISTRIBUTION
        print("🔍 Analyzing Resolution Distribution...")
        resolutions = self.analyze_resolutions()
        # Signals: min_w, max_w, min_h, max_h, aspect_ratios
        # ⚠️ WARNING: If aspect_ratio variance > 1.5x → FIX!
        
        # 2. CHANNEL DRIFT VERIFICATION
        print("🔍 Checking Channel Statistics...")
        channel_stats = self.calculate_channel_stats()
        # Signals: mean_R, mean_G, mean_B, var_R, var_G, var_B
        # ⚠️ WARNING: If channel means vary widely → Normalize!
        
        # 3. IMAGE QUALITY METRICS
        print("🔍 Testing Image Quality...")
        quality_scores = self.check_image_quality()
        # Signals: sharpness, blur_score, contrast, noise_level
        # ❌ FAIL: If blur_score > 150 (Motion blur)
        
        # 4. JPEG COMPRESSION ARTIFACTS
        print("🔍 Detecting Compression Artifacts...")
        compression_level = self.check_compression()
        # Signals: avg_jpeg_quality, blocky_artifacts
        # ⚠️ WARNING: If avg_jpeg_quality < 70 → Recompress!
        
        # 5. PIXEL CLASS REPRESENTATION MATRIX
        print("🔍 Analyzing Class Distribution...")
        class_dist = self.analyze_pixel_distribution()
        # Signals: background_pixels %, target_pixels %
        # ⚠️ WARNING: If target_pixels < 5% → Dice Loss needed!
        
        # 6. FILE INTEGRITY CHECK
        print("🔍 Verifying File Integrity...")
        corrupt_files = self.check_file_integrity()
        # ❌ FAIL: Any corrupt/0-byte files → Remove!
        
        # 7. METADATA DRIFT ANALYSIS
        print("🔍 Analyzing EXIF/Color Profiles...")
        metadata_issues = self.check_metadata()
        # ⚠️ WARNING: Different color profiles → Normalize!
        
    def analyze_resolutions(self):
        """Analyze image dimensions and aspect ratios"""
        import cv2
        import os
        from collections import defaultdict
        
        resolutions = defaultdict(int)
        aspect_ratios = []
        
        for img_file in os.listdir(self.path):
            img = cv2.imread(os.path.join(self.path, img_file))
            h, w = img.shape[:2]
            resolutions[(w, h)] += 1
            aspect_ratios.append(w/h)
            
        return {
            'resolutions': resolutions,
            'min_w': min([r[0] for r in resolutions]),
            'max_w': max([r[0] for r in resolutions]),
            'min_h': min([r[1] for r in resolutions]),
            'max_h': max([r[1] for r in resolutions]),
            'aspect_ratio_std': np.std(aspect_ratios)
        }
    
    def calculate_channel_stats(self):
        """Calculate per-channel mean and variance"""
        import cv2
        import numpy as np
        
        means = {'R': [], 'G': [], 'B': []}
        vars = {'R': [], 'G': [], 'B': []}
        
        for img_file in os.listdir(self.path):
            img = cv2.imread(os.path.join(self.path, img_file))
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            for i, channel in enumerate(['R', 'G', 'B']):
                channel_data = img[:, :, i]
                means[channel].append(np.mean(channel_data))
                vars[channel].append(np.var(channel_data))
        
        return {
            'mean': {c: np.mean(means[c]) for c in means},
            'std': {c: np.std(means[c]) for c in means},
            'variance': {c: np.mean(vars[c]) for c in vars}
        }
    
    def check_image_quality(self):
        """Check image sharpness and blur"""
        import cv2
        import numpy as np
        
        sharpness_scores = []
        
        for img_file in os.listdir(self.path):
            img = cv2.imread(os.path.join(self.path, img_file))
            # Laplacian variance = sharpness
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            sharpness_scores.append(laplacian.var())
        
        return {
            'avg_sharpness': np.mean(sharpness_scores),
            'blur_count': sum(1 for s in sharpness_scores if s < 150),
            'blur_ratio': sum(1 for s in sharpness_scores if s < 150) / len(sharpness_scores)
        }
    
    def check_file_integrity(self):
        """Check for corrupt or 0-byte files"""
        import os
        
        corrupt_files = []
        
        for img_file in os.listdir(self.path):
            file_path = os.path.join(self.path, img_file)
            if os.path.getsize(file_path) == 0:
                corrupt_files.append(file_path)
            else:
                # Try to read image
                try:
                    import cv2
                    img = cv2.imread(file_path)
                    if img is None:
                        corrupt_files.append(file_path)
                except:
                    corrupt_files.append(file_path)
        
        return corrupt_files
```

---

### **🎵 AUDIO DATASETS**

```python
class AudioDatasetAudit:
    """Complete audit checklist for audio datasets"""
    
    def run_full_audit(self):
        """Run all audio dataset checks"""
        
        # 1. SAMPLE RATE CONSISTENCY
        print("🔍 Checking Sample Rate...")
        sample_rates = self.check_sample_rates()
        # ⚠️ WARNING: Different sample rates → Resample!
        
        # 2. SIGNAL-TO-NOISE RATIO (SNR)
        print("🔍 Measuring SNR...")
        snr_values = self.calculate_snr()
        # ❌ FAIL: If SNR < 15dB → High background noise!
        
        # 3. AUDIO CLIPPING RATIO
        print("🔍 Detecting Clipping...")
        clipping_ratio = self.detect_clipping()
        # ❌ FAIL: If clipping_ratio > 5% → Distorted audio!
        
        # 4. SILENCE DURATION DENSITY
        print("🔍 Analyzing Silence...")
        silence_stats = self.analyze_silence()
        # ⚠️ WARNING: If silence > 30% → Trim!
        
        # 5. DURATION DISTRIBUTION
        print("🔍 Analyzing Duration...")
        duration_stats = self.analyze_duration()
        # ⚠️ WARNING: High variance → Trim/Pad!
        
    def check_sample_rates(self):
        """Check audio sample rate consistency"""
        import librosa
        import os
        
        sample_rates = []
        
        for audio_file in os.listdir(self.path):
            try:
                sr = librosa.get_samplerate(os.path.join(self.path, audio_file))
                sample_rates.append(sr)
            except:
                pass
        
        return {
            'unique_rates': list(set(sample_rates)),
            'most_common': max(set(sample_rates), key=sample_rates.count),
            'issues': len(set(sample_rates)) > 1
        }
    
    def detect_clipping(self):
        """Detect audio clipping/distortion"""
        import librosa
        import numpy as np
        
        total_clipped = 0
        total_frames = 0
        
        for audio_file in os.listdir(self.path):
            y, sr = librosa.load(os.path.join(self.path, audio_file))
            clipped = np.sum(np.abs(y) >= 0.95) / len(y)
            total_clipped += clipped
            total_frames += 1
        
        return total_clipped / total_frames
```

---

### **📊 TABULAR/NUMERICAL DATASETS**

```python
class TabularDatasetAudit:
    """Complete audit checklist for tabular datasets"""
    
    def run_full_audit(self, df):
        """Run all tabular dataset checks"""
        
        # 1. MISSING VALUE ANALYSIS
        print("🔍 Scanning Missing Values...")
        missing_stats = self.check_missing_values(df)
        # ⚠️ WARNING: If any feature > 40% missing → Consider dropping
        
        # 2. MULTICOLLINEARITY DIAGNOSTIC
        print("🔍 Checking Multicollinearity...")
        vif_scores = self.calculate_vif(df)
        # ❌ FAIL: If any VIF > 10 → Remove/recombine feature!
        
        # 3. CORRELATION ANALYSIS
        print("🔍 Analyzing Correlations...")
        correlations = self.calculate_correlations(df)
        # ⚠️ WARNING: If correlation > 0.85 → Multicollinearity!
        
        # 4. IMBALANCE DISTRIBUTION
        print("🔍 Checking Class Distribution...")
        imbalance_ratio = self.check_class_imbalance(df)
        # ❌ FAIL: If imbalance_ratio > 10:1 → Use Focal Loss!
        
        # 5. OUTLIER DETECTION
        print("🔍 Detecting Outliers...")
        outliers = self.detect_outliers(df)
        # ⚠️ WARNING: If > 1% outliers → Apply RobustScaler or remove
        
        # 6. CARDINALITY ANALYSIS
        print("🔍 Checking Categorical Cardinality...")
        cardinality = self.check_cardinality(df)
        # ❌ FAIL: If any categorical feature > 100 unique values → Encode appropriately!
        
        # 7. DATA LEAKAGE DETECTION
        print("🔍 Scanning for Data Leakage...")
        leakage_issues = self.detect_data_leakage(df)
        # ❌ FAIL: Any leakage → Fix immediately!
        
    def check_missing_values(self, df):
        """Analyze missing values in dataset"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        return {
            'total_missing': missing.sum(),
            'features_with_missing': list(missing[missing > 0].index),
            'high_missing_features': list(missing_pct[missing_pct > 40].index)
        }
    
    def calculate_vif(self, df):
        """Calculate Variance Inflation Factor for multicollinearity"""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        X = df.select_dtypes(include=[np.number])
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        return vif_data
    
    def check_class_imbalance(self, df, target_col='target'):
        """Check class imbalance ratio"""
        from collections import Counter
        
        counts = df[target_col].value_counts()
        max_class = counts.max()
        min_class = counts.min()
        imbalance_ratio = max_class / min_class
        
        return {
            'class_counts': counts.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'severity': 'HIGH' if imbalance_ratio > 10 else 'MEDIUM' if imbalance_ratio > 5 else 'LOW'
        }
    
    def detect_outliers(self, df, method='IQR'):
        """Detect outliers using IQR or Z-score"""
        outliers = {}
        
        if method == 'IQR':
            for col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        
        return outliers
    
    def detect_data_leakage(self, df):
        """Detect potential data leakage"""
        leakage_issues = []
        
        # Check for target-related columns that shouldn't exist
        # Check for unique identifiers
        # Check for future information in time series
        
        # Example: Check if any column has perfect correlation with target
        correlations = df.corr()['target'].abs()
        perfect_corr = correlations[correlations > 0.99].index.tolist()
        if perfect_corr:
            leakage_issues.append(f"Potentially leaked features: {perfect_corr}")
        
        return leakage_issues
```

---

## 4. **DIAGNOSTIC SIGNALS & THEIR MEANINGS**

| **Signal** | **What It Measures** | **Normal Range** | **Problem Range** | **What It Means** |
|------------|---------------------|------------------|-------------------|-------------------|
| **Imbalance Ratio (ρ)** | Ratio between majority/minority class | 1:1 to 3:1 | > 10:1 | Model will ignore minority class |
| **OOV Rate** | Out-of-vocabulary tokens | < 5% | > 15% | Domain mismatch / New vocabulary |
| **Laplacian Variance** | Image sharpness | > 150 | < 150 | Blurry images → Low quality |
| **SNR** | Signal-to-Noise Ratio | > 25dB | < 15dB | Too much background noise |
| **VIF** | Multicollinearity | < 5 | > 10 | Redundant features → Unstable model |
| **Fleiss' Kappa** | Annotator agreement | > 0.70 | < 0.60 | Labeling inconsistency |
| **FID** | Image quality (generative) | Lower is better | > 100 | Poor generation quality |
| **MMD** | Domain shift | < 0.05 | > 0.15 | Train/Test distribution mismatch |
| **Perplexity** | Language model fit | Low is good | > 100 | Text doesn't fit model well |
| **Duplicate Ratio** | Duplicate samples | < 5% | > 15% | Data redundancy → Overfitting |

---

## 5. **RISK IDENTIFICATION MATRIX**

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         RISK SEVERITY CLASSIFICATION                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🔴 CRITICAL (MUST FIX BEFORE TRAINING)                                    ║
║  ├── Data Leakage (Look-ahead bias)                                        ║
║  ├── Complete class absence in test split                                  ║
║  ├── Corrupted/Unreadable files (>5%)                                      ║
║  └── Mismatched channel dimensions (e.g., 3-channel vs 1-channel)          ║
║                                                                              ║
║  🟠 HIGH (Major performance impact)                                         ║
║  ├── Severe class imbalance (>10:1)                                        ║
║  ├── High outlier ratio (>2%)                                              ║
║  ├── Annotator disagreement (Kappa < 0.60)                                 ║
║  └── Domain shift in test data                                             ║
║                                                                              ║
║  🟡 MEDIUM (Noticeable performance drop)                                    ║
║  ├── Moderate imbalance (5:1 to 10:1)                                      ║
║  ├── Moderate missing values (20-40%)                                      ║
║  ├── JPEG compression artifacts                                            ║
║  └── High cardinality in categorical features                              ║
║                                                                              ║
║  🟢 LOW (Minor impact, can be handled during training)                     ║
║  ├── Slight resolution variations                                          ║
║  ├── Minor color profile differences                                       ║
║  ├── Small OOV rate (5-10%)                                                ║
║  └── Low but acceptable inter-annotator agreement                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 6. **CORRECTION STRATEGIES & CODE**

### **A. Imbalance Correction**

```python
def correct_imbalance(X, y, method='focal_loss'):
    """
    Correct class imbalance using various strategies
    
    Methods:
    - 'focal_loss': Apply Focal Loss (best for DL)
    - 'balanced_sampling': Oversample minority class
    - 'class_weight': Apply class weights
    - 'smote': Synthetic oversampling
    """
    
    if method == 'focal_loss':
        # PyTorch Focal Loss Implementation
        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ce_loss = nn.CrossEntropyLoss(reduction='none')
            
            def forward(self, inputs, targets):
                ce_loss = self.ce_loss(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss()
    
    elif method == 'balanced_sampling':
        from sklearn.utils import resample
        
        # Separate classes
        df = pd.DataFrame(X)
        df['target'] = y
        classes = df['target'].unique()
        
        max_samples = df['target'].value_counts().max()
        balanced_dfs = []
        
        for cls in classes:
            cls_df = df[df['target'] == cls]
            if len(cls_df) < max_samples:
                # Oversample
                cls_df = resample(cls_df, replace=True, n_samples=max_samples, random_state=42)
            balanced_dfs.append(cls_df)
        
        balanced_df = pd.concat(balanced_dfs)
        X_balanced = balanced_df.drop('target', axis=1)
        y_balanced = balanced_df['target']
        
        return X_balanced, y_balanced
    
    elif method == 'class_weight':
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), class_weights))
        
        return class_weight_dict
    
    elif method == 'smote':
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        
        return X_smote, y_smote
```

### **B. Missing Value Correction**

```python
def correct_missing_values(df, method='mice'):
    """
    Handle missing values using various strategies
    
    Methods:
    - 'drop': Drop rows with missing values
    - 'mean': Fill with mean (only for numerical)
    - 'median': Fill with median (robust to outliers)
    - 'mode': Fill with mode (for categorical)
    - 'mice': Multivariate Imputation by Chained Equations (best)
    """
    
    if method == 'drop':
        df_clean = df.dropna()
        return df_clean
    
    elif method == 'mean':
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        return df_clean
    
    elif method == 'median':
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        return df_clean
    
    elif method == 'mode':
        df_clean = df.copy()
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        return df_clean
    
    elif method == 'mice':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        
        imputer = IterativeImputer(max_iter=10, random_state=0)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df),
            columns=df.columns
        )
        return df_imputed
```

### **C. Image Correction**

```python
def correct_images(image_paths, method='aspect_preserve'):
    """
    Correct image issues
    
    Methods:
    - 'aspect_preserve': Maintain aspect ratio with padding
    - 'normalize': Channel normalization
    - 'denoise': Remove noise
    - 'sharpen': Improve blurry images
    - 'recompress': Fix JPEG artifacts
    """
    import cv2
    import numpy as np
    
    if method == 'aspect_preserve':
        def aspect_preserve_resize(img, target_size=(224, 224)):
            h, w = img.shape[:2]
            target_h, target_w = target_size
            
            # Calculate scaling
            scale = min(target_w/w, target_h/h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            resized = cv2.resize(img, (new_w, new_h))
            
            # Pad to target size
            pad_w = (target_w - new_w) // 2
            pad_h = (target_h - new_h) // 2
            
            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
            
            return padded
    
    elif method == 'normalize':
        def normalize_image(img):
            # Channel-wise normalization
            for c in range(3):
                img[:,:,c] = (img[:,:,c] - np.mean(img[:,:,c])) / np.std(img[:,:,c])
            return img
    
    elif method == 'denoise':
        def denoise_image(img):
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    elif method == 'sharpen':
        def sharpen_image(img):
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            return cv2.filter2D(img, -1, kernel)
```

### **D. Domain Shift Correction**

```python
def correct_domain_shift(source_domain, target_domain, method='dann'):
    """
    Correct domain shift between train and test
    
    Methods:
    - 'dann': Domain Adversarial Neural Network
    - 'style_transfer': Neural Style Transfer
    - 'augmentation': Domain-specific augmentation
    """
    
    if method == 'dann':
        # Domain Adversarial Neural Network Implementation
        class DANN(nn.Module):
            def __init__(self, n_features, n_classes, n_domains):
                super().__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Linear(n_features, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )
                
                self.classifier = nn.Linear(128, n_classes)
                self.domain_classifier = nn.Linear(128, n_domains)
            
            def forward(self, x, alpha=1.0):
                features = self.feature_extractor(x)
                
                # For gradient reversal during training
                reversed_features = GradientReversal.apply(features, alpha)
                
                class_output = self.classifier(features)
                domain_output = self.domain_classifier(reversed_features)
                
                return class_output, domain_output
        
        class GradientReversal(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, alpha):
                ctx.alpha = alpha
                return x.view_as(x)
            
            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.neg() * ctx.alpha, None
        
        return DANN
    
    elif method == 'style_transfer':
        # Neural Style Transfer for domain adaptation
        pass
    
    elif method == 'augmentation':
        def domain_augmentation(img, domain_type='medical'):
            if domain_type == 'medical':
                # Medical image specific augmentations
                # Add Gaussian noise, intensity shifts, etc.
                pass
            elif domain_type == 'satellite':
                # Satellite image augmentations
                # Color jitter, rotation, etc.
                pass
```

---

## 7. **MODEL SELECTION GUIDE**

| **Dataset Type** | **Task** | **Recommended Models** | **Why?** |
|------------------|----------|----------------------|----------|
| **Text (Short)** | Classification | BERT, RoBERTa, DistilBERT | Pre-trained on large text corpus |
| **Text (Long)** | Document Classification | Longformer, LED | Handles long sequences |
| **Text (Resumes)** | NER | LayoutLM, SpaCy + BERT | Layout matters in resumes |
| **Text (LLM)** | Generation | GPT-4, Llama, Mistral | Excellent generation capabilities |
| **Image (2D)** | Classification | ResNet, EfficientNet, ViT | Strong feature extraction |
| **Image (Segmentation)** | Medical/Segmentation | UNet, SegNet, DeepLab | Pixel-level accuracy |
| **Image (Object Detection)** | Object Detection | YOLO, Faster R-CNN, DETR | Real-time + accurate |
| **Video** | Action Recognition | I3D, TimeSformer, VideoMAE | Temporal understanding |
| **Audio** | Speech Recognition | Whisper, Wav2Vec, AST | State-of-the-art ASR |
| **Audio** | Music/Genre | CRNN, MusicTransformer | Temporal + spectral features |
| **3D (Point Cloud)** | Segmentation | PointNet, PointTransformer | Handles unordered points |
| **3D (Medical)** | Segmentation | 3D UNet, VoxNet | Volumetric processing |
| **Tabular** | Classification | XGBoost, LightGBM, CatBoost | Best for structured data |
| **Tabular (Small)** | Classification | Random Forest, SVM | Simple, interpretable |
| **Time Series** | Forecasting | LSTM, TCN, PatchTST | Sequential pattern learning |
| **Graph** | Node Classification | GCN, GAT, GraphSAGE | Captures relational structure |
| **Multimodal** | Multi-task | CLIP, LLaVA, Flava | Handles multiple modalities |

---

## 8. **VALIDATION FRAMEWORK**

### **A. Validation Metric Selection**

```python
def select_validation_metrics(task_type, imbalance_info):
    """
    Select appropriate validation metrics based on task and data
    """
    
    metrics = {}
    
    if task_type == 'classification':
        if imbalance_info['imbalance_ratio'] > 5:
            metrics = {
                'primary': 'F1-Macro',
                'secondary': ['Precision', 'Recall', 'ROC-AUC'],
                'monitor': 'Balanced Accuracy'
            }
        else:
            metrics = {
                'primary': 'Accuracy',
                'secondary': ['F1', 'Precision', 'Recall', 'ROC-AUC'],
                'monitor': 'Accuracy'
            }
    
    elif task_type == 'segmentation':
        metrics = {
            'primary': 'Dice Coefficient' if imbalance_info['imbalance_ratio'] > 10 else 'IoU',
            'secondary': ['IoU', 'Pixel Accuracy'],
            'monitor': 'Dice Coefficient'
        }
    
    elif task_type == 'regression':
        metrics = {
            'primary': 'MAE',
            'secondary': ['RMSE', 'R²', 'MAPE'],
            'monitor': 'RMSE'
        }
    
    elif task_type == 'object_detection':
        metrics = {
            'primary': 'mAP@0.5',
            'secondary': ['mAP@0.5:0.95', 'Recall'],
            'monitor': 'mAP'
        }
    
    return metrics
```

### **B. Train/Test Split Strategies**

```python
def get_split_strategy(dataset_type):
    """
    Get appropriate splitting strategy based on dataset type
    """
    
    strategies = {
        'time_series': {
            'method': 'TimeSeriesSplit',
            'description': 'Respects temporal order',
            'code': 'from sklearn.model_selection import TimeSeriesSplit'
        },
        'longitudinal': {
            'method': 'LeaveOneGroupOut',
            'description': 'Same subject stays together',
            'code': 'from sklearn.model_selection import LeaveOneGroupOut'
        },
        'cross_sectional': {
            'method': 'StratifiedShuffleSplit',
            'description': 'Random with class stratification',
            'code': 'from sklearn.model_selection import StratifiedShuffleSplit'
        },
        'image_medical': {
            'method': 'PatientGroupKFold',
            'description': 'Patient IDs in same fold',
            'code': 'from sklearn.model_selection import GroupKFold'
        },
        'video': {
            'method': 'VideoLevelSplit',
            'description': 'Same video stays together',
            'code': 'Custom - Split by video ID'
        },
        'text': {
            'method': 'StratifiedKFold',
            'description': 'Respects class distribution',
            'code': 'from sklearn.model_selection import StratifiedKFold'
        }
    }
    
    return strategies.get(dataset_type, strategies['cross_sectional'])
```

---

## 9. **COMPLETE PYTHON IMPLEMENTATION**

```python
"""
COMPLETE DATASET AUDITING SYSTEM
Enterprise-Grade Data Quality Framework
"""

import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

@dataclass
class AuditConfig:
    """Configuration for dataset audit"""
    dataset_path: str
    dataset_type: str  # 'image', 'text', 'tabular', 'audio', 'video', '3d'
    task_type: str     # 'classification', 'segmentation', 'regression', 'detection'
    target_column: Optional[str] = None
    test_size: float = 0.2
    random_state: int = 42
    verbose: bool = True
    
    # Thresholds
    imbalance_threshold: float = 5.0
    missing_threshold: float = 0.4
    correlation_threshold: float = 0.85
    vif_threshold: float = 10.0
    oov_threshold: float = 0.15

# ============================================================================
# SECTION 2: BASE AUDIT CLASS
# ============================================================================

class DatasetAuditor:
    """Main dataset auditing class"""
    
    def __init__(self, config: AuditConfig):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dataset_audit.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_full_audit(self):
        """Run complete dataset audit"""
        self.logger.info(f"Starting audit for dataset: {self.config.dataset_path}")
        self.logger.info(f"Dataset Type: {self.config.dataset_type}")
        
        audit_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': self.config.dataset_path,
            'dataset_type': self.config.dataset_type,
            'status': 'RUNNING',
            'checks': {},
            'issues': [],
            'recommendations': []
        }
        
        # Route to appropriate auditor
        if self.config.dataset_type == 'image':
            results = self.audit_image_dataset()
        elif self.config.dataset_type == 'text':
            results = self.audit_text_dataset()
        elif self.config.dataset_type == 'tabular':
            results = self.audit_tabular_dataset()
        elif self.config.dataset_type == 'audio':
            results = self.audit_audio_dataset()
        else:
            self.logger.error(f"Unsupported dataset type: {self.config.dataset_type}")
            audit_results['status'] = 'FAILED'
            return audit_results
        
        audit_results['checks'] = results
        audit_results['status'] = 'COMPLETED'
        
        # Generate summary
        summary = self.generate_summary(audit_results)
        audit_results['summary'] = summary
        
        # Save results
        self.save_results(audit_results)
        
        return audit_results
    
    # ========================================================================
    # IMAGE DATASET AUDIT
    # ========================================================================
    
    def audit_image_dataset(self):
        """Audit image dataset"""
        results = {}
        issues = []
        
        self.logger.info("Starting image dataset audit...")
        
        # Get all image files
        image_files = [f for f in os.listdir(self.config.dataset_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            self.logger.error("No image files found!")
            return {'error': 'No images found'}
        
        self.logger.info(f"Found {len(image_files)} images")
        
        # 1. Resolution Analysis
        self.logger.info("Analyzing resolutions...")
        resolutions = []
        aspect_ratios = []
        
        for img_file in image_files[:100]:  # Sample for speed
            try:
                img_path = os.path.join(self.config.dataset_path, img_file)
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                resolutions.append((w, h))
                aspect_ratios.append(w/h)
            except Exception as e:
                self.logger.warning(f"Could not read {img_file}: {e}")
                issues.append(f"Unreadable image: {img_file}")
        
        if resolutions:
            widths = [r[0] for r in resolutions]
            heights = [r[1] for r in resolutions]
            
            results['resolution_analysis'] = {
                'min_width': min(widths),
                'max_width': max(widths),
                'min_height': min(heights),
                'max_height': max(heights),
                'avg_aspect_ratio': np.mean(aspect_ratios),
                'aspect_ratio_std': np.std(aspect_ratios),
                'unique_resolutions': len(set(resolutions)),
                'sample_count': len(resolutions)
            }
            
            # Check for issues
            if results['resolution_analysis']['aspect_ratio_std'] > 0.5:
                issues.append("High aspect ratio variance - consider aspect-preserving resize")
            
            if results['resolution_analysis']['unique_resolutions'] > 10:
                issues.append("Many unique resolutions - consider resizing all images to same size")
        
        # 2. Channel Analysis
        self.logger.info("Analyzing channel statistics...")
        channel_means = {'R': [], 'G': [], 'B': []}
        channel_vars = {'R': [], 'G': [], 'B': []}
        
        for img_file in image_files[:50]:
            try:
                img_path = os.path.join(self.config.dataset_path, img_file)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                for i, channel in enumerate(['R', 'G', 'B']):
                    channel_data = img_rgb[:, :, i]
                    channel_means[channel].append(np.mean(channel_data))
                    channel_vars[channel].append(np.var(channel_data))
            except:
                continue
        
        results['channel_analysis'] = {
            'mean': {c: np.mean(channel_means[c]) for c in channel_means},
            'std': {c: np.std(channel_means[c]) for c in channel_means},
            'variance': {c: np.mean(channel_vars[c]) for c in channel_vars}
        }
        
        # 3. Quality Analysis
        self.logger.info("Analyzing image quality...")
        sharpness_scores = []
        
        for img_file in image_files[:50]:
            try:
                img_path = os.path.join(self.config.dataset_path, img_file)
                img = cv2.imread(img_path)
                laplacian = cv2.Laplacian(img, cv2.CV_64F)
                sharpness_scores.append(laplacian.var())
            except:
                continue
        
        results['quality_analysis'] = {
            'avg_sharpness': np.mean(sharpness_scores) if sharpness_scores else 0,
            'blur_count': sum(1 for s in sharpness_scores if s < 150) if sharpness_scores else 0,
            'blur_ratio': sum(1 for s in sharpness_scores if s < 150) / len(sharpness_scores) if sharpness_scores else 0
        }
        
        if results['quality_analysis']['blur_ratio'] > 0.1:
            issues.append(f"High blur ratio: {results['quality_analysis']['blur_ratio']:.2%} - consider sharpening or removing blurry images")
        
        # 4. Issues summary
        results['issues'] = issues
        results['total_issues'] = len(issues)
        
        self.logger.info(f"Image audit completed. Found {len(issues)} issues.")
        return results
    
    # ========================================================================
    # TABULAR DATASET AUDIT
    # ========================================================================
    
    def audit_tabular_dataset(self):
        """Audit tabular dataset"""
        results = {}
        issues = []
        
        self.logger.info("Starting tabular dataset audit...")
        
        # Load data
        try:
            if self.config.dataset_path.endswith('.csv'):
                df = pd.read_csv(self.config.dataset_path)
            elif self.config.dataset_path.endswith('.xlsx'):
                df = pd.read_excel(self.config.dataset_path)
            elif self.config.dataset_path.endswith('.parquet'):
                df = pd.read_parquet(self.config.dataset_path)
            else:
                df = pd.read_csv(self.config.dataset_path)
            
            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            results['dataset_info'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'memory_usage': df.memory_usage().sum() / 1024**2  # MB
            }
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return {'error': str(e)}
        
        # 1. Missing Values Analysis
        self.logger.info("Analyzing missing values...")
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        high_missing = missing_percentages[missing_percentages > self.config.missing_threshold * 100]
        
        results['missing_analysis'] = {
            'total_missing': missing_counts.sum(),
            'features_with_missing': list(missing_counts[missing_counts > 0].index),
            'high_missing_features': list(high_missing.index),
            'missing_percentages': missing_percentages.to_dict()
        }
        
        if len(high_missing) > 0:
            issues.append(f"Features with > {self.config.missing_threshold*100}% missing values: {list(high_missing.index)}")
        
        # 2. Data Types Analysis
        self.logger.info("Analyzing data types...")
        results['data_types'] = {
            'numerical': list(df.select_dtypes(include=['int64', 'float64']).columns),
            'categorical': list(df.select_dtypes(include=['object', 'category']).columns),
            'datetime': list(df.select_dtypes(include=['datetime64']).columns)
        }
        
        # 3. Cardinality Analysis
        self.logger.info("Analyzing cardinality...")
        cardinality = {}
        for col in df.select_dtypes(include=['object', 'category']).columns:
            unique_count = df[col].nunique()
            cardinality[col] = unique_count
            
            if unique_count > 100:
                issues.append(f"High cardinality in {col}: {unique_count} unique values - consider encoding appropriately")
        
        results['cardinality'] = cardinality
        
        # 4. Correlation Analysis (if target column specified)
        if self.config.target_column and self.config.target_column in df.columns:
            self.logger.info(f"Analyzing correlations with target: {self.config.target_column}")
            
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) > 1:
                correlations = df[numerical_cols].corr()[self.config.target_column]
                high_corr = correlations[abs(correlations) > self.config.correlation_threshold].index.tolist()
                
                results['correlation_analysis'] = {
                    'target': self.config.target_column,
                    'high_correlations': high_corr
                }
                
                if len(high_corr) > 1:  # Exclude target itself
                    issues.append(f"High correlations (> {self.config.correlation_threshold}): {high_corr}")
        
        # 5. Class Imbalance (if classification)
        if self.config.task_type == 'classification' and self.config.target_column:
            self.logger.info("Checking class imbalance...")
            class_counts = df[self.config.target_column].value_counts()
            class_ratios = class_counts / class_counts.sum()
            
            max_ratio = class_counts.max() / class_counts.min()
            
            results['class_imbalance'] = {
                'class_counts': class_counts.to_dict(),
                'class_ratios': class_ratios.to_dict(),
                'imbalance_ratio': max_ratio,
                'num_classes': len(class_counts)
            }
            
            if max_ratio > self.config.imbalance_threshold:
                issues.append(f"Severe class imbalance: ratio = {max_ratio:.2f} - consider Focal Loss or oversampling")
        
        # 6. Outlier Detection
        self.logger.info("Detecting outliers...")
        outliers = {}
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            if outlier_count > 0:
                outliers[col] = outlier_count
        
        results['outliers'] = outliers
        
        # Summary
        results['issues'] = issues
        results['total_issues'] = len(issues)
        
        self.logger.info(f"Tabular audit completed. Found {len(issues)} issues.")
        return results
    
    # ========================================================================
    # TEXT DATASET AUDIT
    # ========================================================================
    
    def audit_text_dataset(self):
        """Audit text dataset"""
        results = {}
        issues = []
        
        self.logger.info("Starting text dataset audit...")
        
        # Try to load as corpus
        text_files = []
        
        if os.path.isdir(self.config.dataset_path):
            text_files = [f for f in os.listdir(self.config.dataset_path) 
                         if f.lower().endswith(('.txt', '.md', '.csv', '.json', '.pdf'))]
        else:
            # Single file
            text_files = [self.config.dataset_path]
        
        if not text_files:
            self.logger.warning("No text files found")
            return {'error': 'No text files found'}
        
        self.logger.info(f"Found {len(text_files)} text files")
        
        # Sample texts
        sample_texts = []
        total_chars = 0
        total_words = 0
        
        for txt_file in text_files[:100]:  # Sample for speed
            try:
                with open(os.path.join(self.config.dataset_path, txt_file), 'r', encoding='utf-8') as f:
                    text = f.read()
                    sample_texts.append(text)
                    total_chars += len(text)
                    total_words += len(text.split())
            except Exception as e:
                self.logger.warning(f"Could not read {txt_file}: {e}")
                issues.append(f"Unreadable text file: {txt_file}")
        
        if sample_texts:
            # Basic statistics
            lengths = [len(t) for t in sample_texts]
            word_counts = [len(t.split()) for t in sample_texts]
            
            results['text_statistics'] = {
                'total_files_sampled': len(sample_texts),
                'avg_char_length': np.mean(lengths),
                'avg_word_count': np.mean(word_counts),
                'min_length': min(lengths),
                'max_length': max(lengths),
                'total_chars': total_chars,
                'total_words': total_words
            }
            
            # Vocabulary analysis
            from collections import Counter
            all_words = ' '.join(sample_texts).split()
            word_freq = Counter(all_words)
            
            results['vocabulary'] = {
                'unique_words': len(word_freq),
                'total_words': len(all_words),
                'top_10_words': word_freq.most_common(10)
            }
            
            # Duplicate detection (approximate)
            from datasketch import MinHash, MinHashLSH
            duplicates_found = 0
            
            for i in range(len(sample_texts)-1):
                for j in range(i+1, min(i+10, len(sample_texts))):
                    if sample_texts[i] == sample_texts[j]:
                        duplicates_found += 1
            
            results['duplicates'] = {
                'duplicate_count': duplicates_found,
                'duplicate_ratio': duplicates_found / (len(sample_texts) * (len(sample_texts)-1) / 2) if len(sample_texts) > 1 else 0
            }
            
            if results['duplicates']['duplicate_ratio'] > 0.1:
                issues.append(f"High duplicate ratio: {results['duplicates']['duplicate_ratio']:.2%}")
        
        # Summary
        results['issues'] = issues
        results['total_issues'] = len(issues)
        
        self.logger.info(f"Text audit completed. Found {len(issues)} issues.")
        return results
    
    # ========================================================================
    # SUMMARY AND REPORTING
    # ========================================================================
    
    def generate_summary(self, audit_results):
        """Generate summary report"""
        summary = {
            'overall_status': 'PASS' if len(audit_results['issues']) == 0 else 'FAIL',
            'total_issues': len(audit_results['issues']),
            'recommendations': []
        }
        
        if len(audit_results['issues']) > 0:
            summary['overall_status'] = 'FAIL'
            
            # Group issues by severity
            critical_issues = []
            high_issues = []
            medium_issues = []
            
            for issue in audit_results['issues']:
                if 'missing' in issue.lower() or 'corrupt' in issue.lower():
                    critical_issues.append(issue)
                elif 'imbalance' in issue.lower() or 'bias' in issue.lower():
                    high_issues.append(issue)
                else:
                    medium_issues.append(issue)
            
            summary['issue_breakdown'] = {
                'critical': critical_issues,
                'high': high_issues,
                'medium': medium_issues
            }
            
            # Generate recommendations
            if critical_issues:
                summary['recommendations'].append("CRITICAL: Fix all critical issues before training")
            if high_issues:
                summary['recommendations'].append("HIGH PRIORITY: Address high-priority issues for better performance")
            
            # Model-specific recommendations
            if self.config.task_type == 'segmentation':
                if any('imbalance' in issue.lower() for issue in audit_results['issues']):
                    summary['recommendations'].append("Use Focal Loss + Dice Loss combination for segmentation")
            
            if self.config.task_type == 'classification':
                if any('imbalance' in issue.lower() for issue in audit_results['issues']):
                    summary['recommendations'].append("Use Focal Loss or class weights for classification")
            
            if self.config.dataset_type == 'image':
                if any('resolution' in issue.lower() for issue in audit_results['issues']):
                    summary['recommendations'].append("Apply aspect-preserving resize with padding")
        
        return summary
    
    def save_results(self, audit_results):
        """Save audit results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"audit_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return str(obj)
            return obj
        
        # Clean results for JSON
        clean_results = json.loads(
            json.dumps(audit_results, default=convert_to_serializable)
        )
        
        with open(output_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}")

# ============================================================================
# SECTION 3: USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Image Dataset Audit
    config = AuditConfig(
        dataset_path='./path/to/images',
        dataset_type='image',
        task_type='segmentation'
    )
    
    auditor = DatasetAuditor(config)
    results = auditor.run_full_audit()
    
    print("\n" + "="*60)
    print("AUDIT SUMMARY")
    print("="*60)
    print(f"Status: {results['summary']['overall_status']}")
    print(f"Total Issues: {results['summary']['total_issues']}")
    
    if results['summary']['total_issues'] > 0:
        print("\nIssues Found:")
        for issue in results['issues']:
            print(f"  - {issue}")
    
    if results['summary']['recommendations']:
        print("\nRecommendations:")
        for rec in results['summary']['recommendations']:
            print(f"  - {rec}")
    
    print("="*60)
    
    # Example 2: Tabular Dataset Audit
    config_tabular = AuditConfig(
        dataset_path='./path/to/data.csv',
        dataset_type='tabular',
        task_type='classification',
        target_column='target'
    )
    
    auditor_tabular = DatasetAuditor(config_tabular)
    results_tabular = auditor_tabular.run_full_audit()
```

---

## 10. **QUICK REFERENCE CHEAT SHEET**

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    QUICK REFERENCE CHEAT SHEET                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📋 STEP 1: DATA TYPE IDENTIFICATION                                         ║
║  ├── Is it structured? → Tabular/CSV                                         ║
║  ├── Is it text? → NLP/LLM                                                   ║
║  ├── Is it images? → CV/CNN                                                  ║
║  ├── Is it audio? → Speech/Music                                             ║ 
║  ├── Is it video? → Action Recognition                                       ║
║  └── Is it 3D? → LiDAR/Medical                                               ║
║                                                                              ║
║  🔍 STEP 2: RUN AUDIT CHECKS                                                ║
║  ├── Missing Values → Drop/Impute                                           ║
║  ├── Class Imbalance → Focal Loss/Weighted Sampling                         ║
║  ├── Corrupt Files → Remove/Re-download                                     ║
║  ├── Data Leakage → Fix split strategy                                      ║
║  └── Domain Shift → Domain Adaptation                                       ║
║                                                                             ║
║  🛠️ STEP 3: APPLY CORRECTIONS                                               ║
║  ├── Imbalance:     Focal Loss, SMOTE, Class Weights                      ║
║  ├── Missing:       MICE Imputation, Drop                                 ║
║  ├── Outliers:      Remove, Clip, RobustScaler                            ║
║  ├── Blurry Images: Sharpen, Remove                                       ║
║  └── Domain Shift:  DANN, Data Augmentation                               ║
║                                                                              ║
║  📊 STEP 4: SPLIT STRATEGY                                                 ║
║  ├── Time Series:   TimeSeriesSplit                                       ║
║  ├── Longitudinal:  LeaveOneGroupOut                                      ║
║  ├── Medical:       GroupKFold (by patient ID)                            ║
║  └── Standard:      StratifiedKFold                                       ║
║                                                                              ║
║  🤖 STEP 5: MODEL SELECTION                                                ║
║  ├── Tabular:       XGBoost, LightGBM, CatBoost                          ║
║  ├── Image 2D:      ResNet, EfficientNet, ViT                            ║
║  ├── Image Seg:     UNet, DeepLab, SegNet                                ║
║  ├── Text:          BERT, RoBERTa, GPT                                   ║
║  ├── Audio:         Whisper, Wav2Vec                                     ║
║  └── Video:         I3D, TimeSformer                                     ║
║                                                                              ║
║  ✅ STEP 6: VALIDATION                                                     ║
║  ├── Classification: F1-Macro (if imbalanced), Accuracy (if balanced)    ║
║  ├── Segmentation:   Dice Score, IoU                                     ║
║  ├── Detection:      mAP                                                 ║
║  └── Regression:     MAE, RMSE                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 📦 **DEPENDENCIES**

```bash
# Install required packages
pip install numpy pandas scikit-learn opencv-python torch torchvision
pip install librosa datasketch statsmodels imbalanced-learn
pip install transformers presidio-analyzer
pip install seaborn matplotlib
```

---


---

**END OF DOCUMENTATION** 📚

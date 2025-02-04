## “No Negatives Needed”: Weakly-Supervised Regression for Tumor Detection in Histopathology 🔬

This repository contains the implementation of the method described in the paper:
“No negatives needed”: weakly-supervised regression for interpretable tumor detection in whole-slide histopathology images". We propose a novel approach using Multiple Instance Learning (MIL) in a regression setting for tumor detection in whole-slide images. This approach offers solution to train tumor detection models while only having positive cases with coarse annotations. Our method leverages **tumor area percentages** as continuous targets, used as a proxy for tumor detection.

### ✨ Key features

- Weakly-Supervised Learning: No need for detailed pixel-level annotations.
- Tumor Area Percentage Regression: Predicts continuous tumor area percentages as a proxy for tumor detection without needing negative cases for training.
- Support of MIL models: Supports various MIL approaches such as mean pooling (MeanPool), attention-based MIL (ABMIL), ([CLAM](https://github.com/mahmoodlab/CLAM)) and ([WeSEG](https://github.com/MarvinLer/tcga_segmentation)).
- Noise Robustness: An analysis of the model's robustness to synthetic noise in tumor percentage labels demonstrated that the models maintain strong performance even when faced with variability and noise in clinical annotations.
- Explainability: Generates attention and logits heatmaps for interpretable AI applications.

### 🛠️ How to Use
Ensure you have Docker installed and running on your system.
1. Clone this repository.

```
git clone https://github.com/DIAGNijmegen/tumor-percentage-MIL-regression.git
cd tumor-percentage-MIL-regression
```

2. Prepare a config file to customize the model training, data paths, optimizer, and other settings. An example of the configuration file is provided in ``config\default.yaml``. 
   - The ``data`` section in the yaml config file contains all the paths to data and results directories. You can find examples of splits and csv_path under ``examples``. ``split_dir`` is the path to the directory containing splits_{n_fold}.csv for each fold, with columns "train", "val", "test" specifying slide IDs for each split. ``features_dirs`` is the path(s) to the pre-computed feature vectors. Multiple directories can be specified (e.g. when training on multiple datasets with features stored in different folders). ``results_dir`` is the ouput folder. ``csv_path`` is the path to a csv file with columns "slide_id" and "label". 

3. Build the Docker image
   
``
docker build -t tumor-detection .
``

4. Run the Docker container, mapping your local directories to the container. Adjust the volume mappings to match your data paths in the config file:
   
``docker run -it -v /path/to/data/:/path/to/data/ -v /path/to/config:/path/to/config tumor-detection /bin/bash
`` 

5. Once inside the Docker container, run the main script using the appropriate configuration file:
python3 main.py --config-name name_config

### 📊 Output and interpretability
The model outputs include:

- Predicted tumor percentages (for MeanPool, ABMIL, CLAM).
- Instance-level predictions.
- Attention scores (for ABMIL and CLAM).
Instance-level predictions and attention scores can be visualized using heatmaps. Heatmap generation relies on code adapted from [CLAM](https://github.com/mahmoodlab/CLAM/blob/f1e93945d5f5ac6ed077cb020ed01cf984780a77/wsi_core/WholeSlideImage.py#L487) to generate heatmaps. To create heatmaps, use the script located in ``utils/generate_heatmaps.py``, which expects a configuration file similar to ``config-heatmaps/default.yaml``.

### 🧪 Testing
To test an existing checkpoint or evaluate the model:

- Specify the ckpt_path in the configuration file under the testing section.
- Set ``only_testing: True`` to skip training and directly evaluate.

### 💡 Tips for Best Results
To improve tumor detection performance and interpretability, consider using the _amplification technique_ introduced in our paper. This method was especially effective in:

Datasets with biopsies or lymph nodes.
Scenarios with small lesions or tumor-free cases, where differentiating between very small tumor percentages and negative slides is challenging.
The amplification technique involves applying a transformation to the tumor area percentage targets during training. In our experiments, a fifth-root transformation was effective. However, you are encouraged to explore transformations that best suit your dataset.

### 📜 Additional scripts
- ``utils/fold_splits.py``: Generates 5-fold splits with stratification on continuous targets.
- ``utils/add_noise.py``: Adds uniform noise to training targets to simulate variability in clinical practice.
- ``utils/generate_heatmaps.py``: Creates heatmaps for instance-level predictions and attention scores.
  
### 💬 Feedback & Support
Feel free to reach out with questions, suggestions, or issues via the repository's Issues tab or contact the authors directly. 

### 🏆 Acknowledgments
This work builds upon contributions from the following repositories:
- [hs2p](https://github.com/clemsgrs/hs2p): Used for patch extraction.
- [CLAM](https://github.com/mahmoodlab/CLAM): Used for feature extraction, heatmap generation, and CLAM model.
- [TCGA Segmenttion](https://github.com/MarvinLer/tcga_segmentation): Starting point for developing the WeSEG model.

This project has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 945358. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation program and EFPIA ([www.imi.europe.eu](\url{www.imi.europe.eu})).

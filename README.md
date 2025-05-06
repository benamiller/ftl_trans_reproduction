# FTL-Trans: Reproduction

This repository contains code and instructions to reproduce the results of the FTL-Trans model for the in-hospital mortality prediction task on the MIMIC-III dataset, as presented in the paper:

> Zhang, D., Thadajarassiri, J., Sen, C., & Rundensteiner, E. (2020). Time-Aware Transformer-based Network for Clinical Notes Series Prediction. _Proceedings of the 5th Machine Learning for Healthcare Conference_, PMLR 126:1-22. [(Link)](https://proceedings.mlr.press/v126/zhang20c.html)

The original FTL-Trans model introduced a hierarchical, time-aware architecture using ClinicalBERT and a custom Flexible Time-aware LSTM (FT-LSTM) to effectively model sequences of clinical notes.

**Original Code:** [https://github.com/zdy93/FTL-Trans](https://github.com/zdy93/FTL-Trans)

**Note on Reproducibility:** The original codebase was developed circa 2020. Running it in modern environments (e.g., Python 3.10+, NumPy 2.0+, Pandas 2.0+) requires specific dependency version management and minor code patches, which are detailed in the steps below. Performance bottlenecks related to data loading and CPU processing within the training loop were also observed and mitigation strategies are discussed.

## Environment Setup

It is highly recommended to use a dedicated virtual environment (e.g., conda).

1. **Python:** Python 3.11 was used in the reproduction environment (issues were encountered with older libraries under Python 3.10+).
    
    Bash
    
    ```
    # Example conda environment creation
    # conda create -n ftltrans python=3.11 pip
    # conda activate ftltrans
    ```
    
2. **Core Dependencies:** Install PyTorch compatible with your CUDA version first. Then install the specific (often older) versions required by the original code, plus necessary upgrades for compatibility fixes.
    
    Bash
    
    ```
    # Install PyTorch (example for CUDA 11.8 - check pytorch.org for your setup)
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install specific versions from original requirements + necessary libraries
    pip install pandas==2.0.0 # Or newer - Requires concat fix in preprocessing.py
    pip install numpy==2.0.0 # Or newer - Requires unicode_ fix in utils.py
    pip install pytorch-transformers==1.2.0
    pip install pytorch-pretrained-bert==0.6.2 # For BertAdam
    pip install tqdm==4.37.0
    pip install dotmap # Install initially, then upgrade below
    pip install six==1.13.0
    pip install matplotlib==3.1.1
    pip install scikit-learn # For KFold if used (though not in main reproduction path here)
    pip install nibabel # Needed by ADNI example, maybe by utils? Check if needed. (Likely not needed for FTL-Trans)
    
    # Upgrade dotmap to fix MutableMapping import error with Python 3.10+
    pip install -U dotmap
    ```
    
    _Note: You might need to adjust PyTorch/CUDA versions for your specific hardware._
    
3. **Hardware:**
    
    - **GPU:** Required for reasonable training times. NVIDIA T4 (16GB VRAM) or A100 (40GB+ VRAM) recommended.
    - **System RAM:** **High RAM is strongly recommended (e.g., >25GB)**, especially for the data preprocessing (concatenation/splitting) phase. Standard Colab (~12GB) is likely insufficient.
    - **Storage:** Sufficient space for MIMIC-III CSVs (~25GB+), intermediate files, and model checkpoints.

## Data Preparation

1. **Download MIMIC-III Data:**
    
    - Obtain access to the MIMIC-III Clinical Database (v1.4 recommended) from PhysioNet: [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/)
    - Download the core CSV files, particularly `ADMISSIONS.csv` and `NOTEEVENTS.csv`.
    - Place them in a designated directory, e.g., `./mimic_raw_data/`.
2. **Download Pre-trained ClinicalBERT:**
    
    - Download the ClinicalBERT checkpoint files (specifically `pytorch_model.bin`, `bert_config.json`, `vocab.txt`) from the repository mentioned in the FTL-Trans paper: [https://github.com/kexinhuang12345/clinicalBERT](https://github.com/kexinhuang12345/clinicalBERT) (Look for download links/instructions there).
    - Create a directory, e.g., `./pretraining/`, and place these three files inside it.
3. **Generate `original_data.csv` (Initial Processing):**
    
    - Use the provided `prepare_ftl_trans_data.py` script (or adapt it) to combine `ADMISSIONS.csv` and `NOTEEVENTS.csv`, extract relevant columns, perform basic cleaning, and assign the mortality label based on `HOSPITAL_EXPIRE_FLAG`.
    - **Crucially, ensure this script outputs a COMMA-separated CSV file (`original_data.csv`)** with the columns: `Adm_ID,Note_ID,chartdate,charttime,TEXT,Label`.
    - Example (assuming script is in `./scripts/` and raw data in `./mimic_raw_data/`):
        
        Bash
        
        ```
        # Make sure prepare_ftl_trans_data.py saves as CSV with ',' separator
        mkdir -p ./data_intermediate/
        python ./scripts/prepare_ftl_trans_data.py \
            --admissions_path ./mimic_raw_data/ADMISSIONS.csv \
            --notes_path ./mimic_raw_data/NOTEEVENTS.csv \
            --output_dir ./data_intermediate/ # Script should save original_data.csv here
        ```
        
4. **Tokenize Data and Split Train/Val/Test (Using Modified `preprocessing.py` Logic):**
    
    - **Code Patches Required:**
        - **`preprocessing.py`:** Modify the script to replace the deprecated `df.append()` call with the `pd.concat()` approach (collect chunks in a list, concatenate after the loop - see `concat_split_code_cell.py` for an example).
        - **`utils.py`:** Modify `pad_sequences` function to replace `np.unicode_` with `np.str_`.
        - **`modeling_patient.py`:** Modify `map_elapse_time` function within `FTLSTM` class to handle `t=0` gracefully, e.g., using `safe_t = torch.max(t, torch.tensor(1e-6).to(t.device))`.
    - **Option A: Run Full (Patched) `preprocessing.py` (Requires High-RAM):**
        
        - Ensure the temporary directory (e.g., `./temp_processed/`) is empty first.
        - Run the patched script:
        
        Bash
        
        ```
        # Ensure temp_dir and output_dir exist and temp_dir is empty
        mkdir -p ./temp_processed/ ./final_split_data/ ./logs/
        rm -rf ./temp_processed/*
        
        # Run patched preprocessing.py (use High-RAM instance)
        python ./scripts/preprocessing.py \
            --original_data "./data_intermediate/original_data.csv" \
            --output_dir "./final_split_data/" \
            --temp_dir "./temp_processed/" \
            --task_name "Mortality_Preprocessing" \
            --log_path "./logs/preprocessing_log.txt" \
            --id_num_neg 5287 `# Adjust as needed` \
            --id_num_pos 5287 `# Adjust as needed` \
            --random_seed 42 \
            --bert_model "./pretraining/" `# Path to ClinicalBERT vocab.txt`
        ```
        
    - **Option B: Use Existing Temp Files (If Step A's first loop completed):**
        
        - Use the provided `concat_split_code_cell.py` (or adapt it) which reads existing `Processed_*.csv` files from `./temp_processed/`, concatenates them, performs the train/val/test split, and saves the results to `./final_split_data/`. This avoids re-running the slow tokenization part but still requires High-RAM for concatenation/splitting.
        
        Bash
        
        ```
        # Ensure output_dir exists
        mkdir -p ./final_split_data/ ./logs/
        
        # Run the concat/split script (use High-RAM instance)
        python ./scripts/concat_split_code_cell.py \
            --temp_dir "./temp_processed/" \
            --output_dir "./final_split_data/" \
            --log_path "./logs/concat_split_log.txt" \
            --id_num_neg 5287 `# Adjust as needed` \
            --id_num_pos 5287 `# Adjust as needed` \
            --num_chunks 185 `# Or determine dynamically` \
            --random_seed 42
        ```
        
    - **Output:** This step produces `train.csv`, `val.csv`, `test.csv` in the specified output directory (e.g., `./final_split_data/`). These files contain processed, tokenized notes, split by patient ID, but notes are _not yet chunked_.
5. **Split Notes into Chunks (`split_into_chunk.py`):**
    
    - Takes the output from the previous step and splits long note sequences into chunks.
    - Run the script:
        
        Bash
        
        ```
        # Ensure output_dir exists
        mkdir -p ./chunked_data/ ./logs/
        
        python ./scripts/split_into_chunk.py \
            --data_dir "./final_split_data/" `# Input dir` \
            --train_data "train.csv" \
            --val_data "val.csv" \
            --test_data "test.csv" \
            --log_path "./logs/split_into_chunk_log.txt" \
            --output_dir "./chunked_data/" `# Output dir` \
            --max_seq_length 128
        ```
        
    - **Output:** Produces the final `train.csv`, `val.csv`, `test.csv` in `./chunked_data/` where each row represents a single chunk. This is the input for the training script.

## Training and Evaluation

1. **Prerequisites:**
    
    - Ensure all data preparation steps above are complete.
    - Ensure the **chunked** data resides in a known directory (e.g., `./chunked_data/`).
    - Ensure the pre-trained ClinicalBERT model files reside in a known directory (e.g., `./pretraining/`).
    - **Code Patches:** Confirm the necessary patches have been applied (especially the `t=0` fix in `modeling_patient.py`'s `map_elapse_time` function and the `np.unicode_` fix in `utils.py`).
    - **Runtime:** Use a **GPU runtime** (T4 or A100 recommended). High System RAM is still advisable for data loading within the script.
2. **Performance Note (Optional but Recommended):** To avoid potential I/O bottlenecks during training, copy the `./chunked_data/` and `./pretraining/` directories to the local Colab VM storage (e.g., `/content/`) before running training, and adjust the paths in the command below accordingly.
    
    Bash
    
    ```
    # Example Copy Commands (run before training command)
    # mkdir -p /content/chunked_data_local/ /content/pretraining_local/
    # echo "Copying chunked data..."
    # cp -r ./chunked_data/* /content/chunked_data_local/
    # echo "Copying pre-trained model..."
    # cp -r ./pretraining/* /content/pretraining_local/
    # echo "Copying finished."
    # --- Then use /content/chunked_data_local/ and /content/pretraining_local/ below ---
    ```
    
3. **Run Training & Evaluation:** Execute `run_clbert_ftlstm.py` with appropriate arguments.
    
    Bash
    
    ```
    # Ensure output and log directories exist
    mkdir -p ./exp_FTL-Trans/ ./logs/
    
    # Change directory to ensure local imports work
    cd ./scripts/ && \
    
    # Run the main training script (ON GPU RUNTIME)
    python run_clbert_ftlstm.py \
        --data_dir "../chunked_data/" `# Adjust if copied locally, e.g., /content/chunked_data_local/` \
        --train_data "train.csv" \
        --val_data "val.csv" \
        --test_data "test.csv" \
        --log_path "../logs/run_clbert_ftlstm_log.txt" `# Log kept in persistent storage` \
        --bert_model "../pretraining/" `# Adjust if copied locally, e.g., /content/pretraining_local/` \
        --embed_mode "all" \
        --task_name "FTL-Trans_Mortality_Reproduction" \
        --max_seq_length 128 `# MUST match split_into_chunk.py` \
        --train_batch_size 32 \
        --eval_batch_size 1 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --warmup_proportion 0.1 \
        --max_chunk_num 64 `# Recommended for A100/T4 to match paper` \
        --seed 42 \
        --gradient_accumulation_steps 1 \
        --output_dir "../exp_FTL-Trans/" `# Output kept in persistent storage` \
        --save_model True
    
    # Change back directory if needed
    cd ..
    ```
    

## Expected Results

- **Logs:** Detailed logs for each script will be saved to the paths specified by `--log_path`. The training log (`run_clbert_ftlstm_log.txt`) will contain epoch timings, training loss, and validation accuracy per epoch.
- **Model Checkpoints:** If `--save_model True` is used, checkpoints (`bert_fine_tuned_with_note_checkpoint_*.pt`) and a final state dictionary (`bert_fine_tuned_with_note_state_dict.pt`) will be saved in the `--output_dir` (e.g., `./exp_FTL-Trans/`).
- **Plots:** A plot of the training loss (`bert_fine_tuned_with_note_training_loss.png`) will be saved in the `--output_dir`.
- **Predictions:** Test set predictions (`test_predictions.csv`) will be saved in the `--output_dir`.
- **Performance Metrics:** Final performance metrics (AUROC, Accuracy, AUPR, etc.) will be printed to the console and saved in the log file. Results should be comparable to those reported in Table 3 of the original paper for FTL-Trans on the mortality task, but allow for minor variations due to environment and randomness.

## Code Structure

- `prepare_ftl_trans_data.py`: Custom script for initial conversion of MIMIC-III CSVs to `original_data.csv`.
- `scripts/preprocessing.py`: Handles tokenization, train/val/test splitting (requires patches).
- `scripts/split_into_chunk.py`: Splits processed notes into fixed-length chunks.
- `scripts/run_clbert_ftlstm.py`: Main script for training and evaluating the FTL-Trans model (requires patches).
- `scripts/modeling_patient.py`: Defines `PatientLevelEmbedding`, `FTLSTM` cell, `FTLSTMLayer` (requires patch).
- `scripts/modeling_readmission.py`: Defines `BertModel` and other BERT components (older style).
- `scripts/other_func.py`: Helper functions for data processing, evaluation.
- `scripts/utils.py`: Utility functions like `pad_sequences` (requires patch).
- `scripts/file_utils.py`: Utilities likely for downloading/caching pre-trained models (used by `modeling_readmission.py`).
- `scripts/concat_split_code_cell.py` (Optional): Alternative script combining concatenation and splitting from existing temporary files.

## Troubleshooting Notes

- **`ModuleNotFoundError`:** Ensure all required libraries (including specific versions like `pytorch_pretrained_bert==0.6.2` and an updated `dotmap`) are installed in the correct environment. Ensure all script files (`modeling_*.py`, `other_func.py`, `utils.py`, `file_utils.py`) are in the same directory from which `run_*.py` scripts are executed (or use `cd`).
- **`AttributeError: ... no attribute 'append'`:** Your pandas version is >= 2.0. Patch `preprocessing.py` to use `pd.concat` instead of `df.append`.
- **`AttributeError:`np.unicode_`was removed...`:** Your NumPy version is >= 2.0. Patch `utils.py` (in `pad_sequences`) to use only `np.str_`.
- **`ImportError: cannot import name 'MutableMapping' from 'collections'`:** Your Python version is >= 3.10. Upgrade `dotmap` using `pip install -U dotmap`.
- **`ValueError: Temp Output directory ... already exists and is not empty`:** Clear the specified temporary directory before re-running `preprocessing.py` OR modify the script to skip this check if you intend to reuse existing temporary files for concatenation/splitting only.
- **Training Stuck (Loss ~0.693, Acc ~0.5):** This likely indicates a numerical stability issue. Ensure the `t=0` fix (`safe_t = torch.max(t, epsilon)`) is correctly applied in `map_elapse_time` within `modeling_patient.py`.
- **Slow Performance / Spiky GPU Utilization:** Likely an I/O bottleneck if reading data from network storage (e.g., Google Drive). Copy data (`chunked_data/`, `pretraining/`) to local VM storage (`/content/`) before running `run_clbert_ftlstm.py`. If utilization is still spiky, it may indicate a CPU bottleneck within the training loop's data preparation steps.

## Citation

Code snippet

```
@InProceedings{pmlr-v126-zhang20c,
 title = 	 {Time-Aware Transformer-based Network for Clinical Notes Series Prediction},
 author = 	 {Zhang, Dongyu and Thadajarassiri, Jidapa and Sen, Cansu and Rundensteiner, Elke},
 booktitle = 	 {Proceedings of the 5th Machine Learning for Healthcare Conference},
 pages = 	 {1--22},
 year = 	 {2020},
 editor = 	 {Flores, Gerardo and Chen, George H. and Cheng, Leo Anthony Celi and Maher, Tristan and Szolovits, Peter and Doshi-Velez, Finale},
 volume = 	 {126},
 series = 	 {Proceedings of Machine Learning Research},
 month = 	 {06--08 Aug},
 publisher = 	 {PMLR},
 pdf = 	 {http://proceedings.mlr.press/v126/zhang20c/zhang20c.pdf},
 url = 	 {https://proceedings.mlr.press/v126/zhang20c.html}
}
```

## Author

Benjamin Miller (Reproduction Study)

--------------------------
# Original Paper README.md

# FTL-Trans
This repository is the official implementation of the **MLHC2020** paper [Time-Aware Transformer-based Network for Clinical Notes Series Prediction](https://proceedings.mlr.press/v126/zhang20c.html). 

## Requirement
### Language
* Python3 == 3.x.x
### Module
* torch==1.3.1+cu92
* pytorch-pretrained-bert==0.6.2
* pytorch-transformers==1.2.0
* tqdm==4.37.0
* dotmap==1.3.8
* six==1.13.0
* matplotlib==3.1.1
* numpy==1.17.3
* pandas==0.25.3
## Dataset
We use [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/). We refer users to the link for requesting access. You can also use some other clinical notes or non-clinical documents as input.

File system expected:
```Linux
data/
  test.csv
  train.csv
  val.csv
```
## Pretrained Model
In our paper, we initialize the transformer layer with [ClinicalBERT](https://github.com/kexinhuang12345/clinicalBERT). We refer user to the link for requesting pre-trained model. You can also use some other pre-trained models, like [BERT](https://github.com/huggingface/transformers).
## Model Prediction
Below list the scripts for running prediction. The file [run_clbert_ftlstm.py](./run_clbert_ftlstm.py) contains the code for the FT-Trans. Other files named as run_\[model\].py contain codes for baseline models.
```cmd
python3 run_clbert_ftlstm.py
  --data_dir ./data_01
  --train_data train.csv
  --val_data val.csv
  --test_data test.csv
  --log_path ./log.txt
  --bert_model ./pretraining
  --embed_mode all
  --task_name FTL-Trans_Prediction
  --max_seq_length 128
  --train_batch_size 32
  --eval_batch_size 1
  --learning_rate 2e-5
  --num_train_epochs 3
  --warmup_proportion 0.1
  --max_chunk_num 32
  --seed 42
  --gradient_accumulation_steps 1
  --output_dir ./exp_FTL-Trans
  --save_model True
```
We refer users to [run_clbert_ftlstm.py](./run_clbert_ftlstm.py) for detalied explanation of each parameter.

We also provide [preprocessing.py](./preprocessing.py) and [split_into_chunk.py](./split_into_chunk.py) for preprocessing data and spliting data into chunks. However, if your data does not has the same format as ours, which means that your data does not have the columns that we have (Adm_ID, Note_ID, chartdate, charttime, TEXT, Label). You need to modify the code before implmenting preprocessing. 

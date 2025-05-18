#!/bin/bash

tissues=(
"Adipose_Subcutaneous"
"Adipose_Visceral_Omentum"
"Adrenal_Gland"
"Artery_Aorta"
"Artery_Coronary"
"Artery_Tibial"
"Brain_Amygdala"
"Brain_Anterior_cingulate_cortex_BA24"
"Brain_Caudate_basal_ganglia"
"Brain_Cerebellar_Hemisphere"
"Brain_Cerebellum"
"Brain_Cortex"
"Brain_Frontal_Cortex_BA9"
"Brain_Hippocampus"
"Brain_Hypothalamus"
"Brain_Nucleus_accumbens_basal_ganglia"
"Brain_Putamen_basal_ganglia"
"Brain_Spinal_cord_cervical_c-1"
"Brain_Substantia_nigra"
"Breast_Mammary_Tissue"
"Cells_Cultured_fibroblasts"
"Cells_EBV-transformed_lymphocytes"
"Colon_Sigmoid"
"Colon_Transverse"
"Esophagus_Gastroesophageal_Junction"
"Esophagus_Mucosa"
"Esophagus_Muscularis"
"Heart_Atrial_Appendage"
"Heart_Left_Ventricle"
"Kidney_Cortex"
"Liver"
"Lung"
"Minor_Salivary_Gland"
"Muscle_Skeletal"
"Nerve_Tibial"
"Ovary"
"Pancreas"
"Pituitary"
"Prostate"
"Skin_Not_Sun_Exposed_Suprapubic"
"Skin_Sun_Exposed_Lower_leg"
"Small_Intestine_Terminal_Ileum"
"Spleen"
"Stomach"
"Testis"
"Thyroid"
"Uterus"
"Vagina"
"Whole_Blood"
)

output_dir=$1
model_path=$2
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

tissue_index=0
while [ $tissue_index -lt ${#tissues[@]} ]; do
  retrying=0
  tissue="${tissues[$tissue_index]}"
  # 检查是否已经处理过该tissue
  if [ -f "$output_dir/$tissue.npy" ]; then
    echo "Skipping tissue: $tissue (already processed)"
    tissue_index=$((tissue_index + 1))
  else
    echo "Processing tissue: $tissue"
    python expr_variant.py --tissue_name "$tissue" --output_dir "$output_dir" --model_name_or_path "$model_path"> "$output_dir/$tissue.log"

    # 检查 Python 脚本是否成功完成
    if [ -f "$output_dir/$tissue.npy" ]; then
      tissue_index=$((tissue_index + 1))
    else
      echo "Failed to process tissue: $tissue, retrying..." # 不增加索引，继续重试当前组织
      retrying=$((retrying + 1))
      if [ $retrying -ge 3 ]; then
        echo "Failed to process tissue: $tissue after 3 retries, skipping..."
        tissue_index=$((tissue_index + 1))
      fi
    fi
  fi
done
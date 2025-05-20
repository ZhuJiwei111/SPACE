import csv
import os
from collections import defaultdict


def parse_fasta(file_path):
    """解析 FASTA 文件，提取染色体、位置、类别和序列信息"""
    sequences = []
    with open(file_path, "r") as file:
        current_header = None
        current_sequence = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    sequences.append((current_header, current_sequence))
                current_header = line[1:]  # 去掉 '>'
                current_sequence = ""
            else:
                current_sequence += line
        if current_header:
            sequences.append((current_header, current_sequence))
    return sequences


def save_to_csv(sequences, output_file):
    """将解析后的数据保存为 CSV 文件"""
    fieldnames = ["chromosome", "start", "end", "category", "sequence"]
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()  # 写入表头
        for header, sequence in sequences:
            try:
                # 解析描述行
                parts = header.split("|")
                if len(parts) != 2:
                    print(f"警告：描述行格式不正确，跳过该记录: {header}")
                    continue
                chromosome_positions, category = parts
                chromosome, positions = chromosome_positions.split(":")
                start, end = positions.split("-")
                # 写入 CSV
                writer.writerow(
                    {
                        "chromosome": chromosome,
                        "start": int(start),
                        "end": int(end),
                        "category": int(category),
                        "sequence": sequence,
                    }
                )
            except Exception as e:
                print(
                    f"错误：解析描述行时发生异常，跳过该记录: {header}，错误信息: {e}"
                )


def process_directory(input_root, output_root):
    """遍历目录结构，处理所有 .fna 文件并保存为 CSV"""
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(".fna"):
                # 输入文件路径
                input_file = os.path.join(root, file)
                # 输出文件路径（保持目录结构）
                relative_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, file.replace(".fna", ".csv"))
                # 解析 FASTA 文件并保存为 CSV
                sequences = parse_fasta(input_file)
                save_to_csv(sequences, output_file)
                print(f"处理完成: {input_file} -> {output_file}")


# 示例使用
input_root = "nucleotide_transformer_downstream_tasks_revised"  # 替换为你的输入文件夹路径
output_root = "./experiments/downstream/dataset"  # 替换为你的输出文件夹路径

# 处理所有文件
process_directory(input_root, output_root)

print("所有文件处理完成！")
